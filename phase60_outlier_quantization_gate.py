"""Phase 60: offline INT4 Q15 single-group outlier peeling and clipping gate.

No COM3, firmware changes, inference, token selection, or model modification.
This replays one previously captured ESP32 activation vector against pretrained
INT4 down-projection rows. All outlier choices use DEVICE activations only.
"""
import argparse
import hashlib
import json
import mmap
from pathlib import Path
import numpy as np

ROWS, COLS, GROUP = 896, 4864, 128
NG = COLS // GROUP
ATOL = RTOL = 0.001
EXPECTED_SHA = '9cb8e2b5e08ab6594f32c3dfd255c96e4ba52e7d6b02586c65cfc928d019ce43'


def q15(v, clip=None):
    a = np.asarray(v, np.float32)
    maximum = np.float32(np.max(np.abs(a)) if clip is None else clip)
    if maximum == 0:
        return np.zeros(a.shape, dtype=np.int16), np.float32(1)
    inv = np.float32(np.float32(32760) / maximum)
    scale = np.float32(maximum / np.float32(32760))
    q = np.clip(np.rint(a * inv), -32760, 32760).astype(np.int16)
    return q, scale


def quantize_all(x):
    q = np.empty((NG, GROUP), np.int16)
    steps = np.empty(NG, np.float32)
    for g in range(NG):
        q[g], steps[g] = q15(x[g*GROUP:(g+1)*GROUP])
    return q, steps


def choose_group(x, q, steps):
    residuals = x.reshape(NG,GROUP) - q.astype(np.float32) * steps[:,None]
    scores = np.sum(np.abs(residuals.astype(np.float64)), axis=1)
    return int(np.argmax(scores)), scores


def create_candidates(x, original_q, original_steps, group, peel_counts=(1, 2, 4), percentiles=(99, 95, 90)):
    """Only one group's 128 Q15 entries change; indices ranked by device abs value."""
    v = x[group*GROUP:(group+1)*GROUP]
    order = np.argsort(-np.abs(v), kind='stable')
    result = [{'name':'baseline', 'q':original_q[group].copy(), 'step':original_steps[group], 'indices':np.empty(0,np.int64), 'values':np.empty(0,np.float32), 'kind':'baseline'}]
    for k in peel_counts:
        if not 1 <= k < GROUP:
            raise ValueError(f'Invalid peeled count: {k}')
        indices = order[:k].copy()
        masked = v.copy()
        masked[indices] = 0
        q,step = q15(masked)
        result.append({'name':f'peel_top_{k}', 'q':q,'step':step,'indices':indices,'values':v[indices].copy(),'kind':'peel'})
    for pct in percentiles:
        if not 0 < pct <= 100: raise ValueError('Invalid percentile')
        limit = np.float32(np.percentile(np.abs(v), pct))
        q,step = q15(v, clip=limit)
        result.append({'name':f'clip_p{pct}', 'q':q,'step':step,'indices':np.empty(0,np.int64),'values':np.empty(0,np.float32),'kind':'clip'})
    return result, order


def unpack_chunk(mm, offset, start, end, row_bytes):
    raw = np.frombuffer(mm[offset+start*row_bytes:offset+end*row_bytes], np.uint8)
    expected = (end-start)*row_bytes
    if raw.size != expected: raise ValueError(f'INT4 row span short: {start}:{end}, {raw.size}/{expected}')
    a = raw.reshape(end-start,row_bytes)
    low = (a[:,:COLS//2]&15).astype(np.int8)
    high = (a[:,:COLS//2]>>4).astype(np.int8)
    low[low>=8]-=16; high[high>=8]-=16
    w = np.empty((end-start,COLS),np.int8)
    w[:,::2] = low; w[:,1::2] = high
    scales = np.frombuffer(a[:,COLS//2:COLS//2+4].copy().tobytes(),dtype='<f4')
    return w,scales


def metric(output, reference):
    error = np.abs(output.astype(np.float64)-reference.astype(np.float64))
    threshold = ATOL + RTOL * np.abs(reference.astype(np.float64))
    return {'mixed_failures':int(np.count_nonzero(error>threshold)),
            'max_abs':float(np.max(error)),
            'mean_abs':float(np.mean(error))}


def scan(mm, layout, device, reference, candidates, group, base_q, base_steps, batch=16):
    offset, rows, cols, row_bytes = layout
    if (rows,cols)!=(ROWS,COLS) or row_bytes<COLS//2+4: raise ValueError('Unexpected INT4 layout')
    reference_output = np.empty(ROWS,np.float32)
    device_exact_output = np.empty(ROWS,np.float32)
    outputs = {c['name']: np.empty(ROWS,np.float32) for c in candidates}
    first = group*GROUP; last=first+GROUP
    for start in range(0,ROWS,batch):
        end=min(start+batch,ROWS)
        w,scales=unpack_chunk(mm,offset,start,end,row_bytes)
        reference_output[start:end]=(w.astype(np.float64)@reference.astype(np.float64)).astype(np.float32)*scales
        device_exact_output[start:end]=(w.astype(np.float64)@device.astype(np.float64)).astype(np.float32)*scales
        acc = np.zeros(end-start,np.float32)
        for g in range(NG):
            dot = w[:,g*GROUP:(g+1)*GROUP].astype(np.int32) @ base_q[g].astype(np.int32)
            acc += dot.astype(np.float32)*base_steps[g]
        original = (w[:,first:last].astype(np.int32)@base_q[group].astype(np.int32)).astype(np.float32)*base_steps[group]
        for cand in candidates:
            if cand['kind']=='baseline':
                outputs[cand['name']][start:end]=acc*scales
                continue
            substitute = (w[:,first:last].astype(np.int32) @ cand['q'].astype(np.int32)).astype(np.float32)*cand['step']
            # Sparse peeled entries require k extra scalar INT4 weight reads and FP32 multiply-adds.
            # No second 128-element dot product. This is NOT a proven zero-overhead change.
            if cand['indices'].size:
                sparse = (w[:,first+cand['indices']].astype(np.float64) @ cand['values'].astype(np.float64)).astype(np.float32)
                substitute += sparse
            outputs[cand['name']][start:end]=(acc-original+substitute)*scales
    return reference_output,device_exact_output,outputs


def run(args):
    metadata=json.loads(Path(args.metadata).read_text(encoding='utf-8'))
    image=Path(args.image)
    if image.stat().st_size != int(metadata['image_size']): raise ValueError('INT4 image size does not match metadata')
    if args.verify_sha:
        with image.open('rb') as f:
            if hashlib.file_digest(f,'sha256').hexdigest()!=EXPECTED_SHA: raise ValueError('INT4 image checksum mismatch')
    m=metadata['layers'][2]['matrices'][6]
    layout=tuple(int(m[k]) for k in ('offset','rows','cols','row_bytes'))
    with np.load(args.trace,allow_pickle=False) as data:
        device=np.asarray(data['device_ACT'],np.float32).copy()
        reference=np.asarray(data['reference_ACT'],np.float32).copy()
    if device.shape!=(COLS,) or reference.shape!=(COLS,): raise ValueError('Expected device_ACT/reference_ACT with 4864 features each')
    if not np.isfinite(device).all() or not np.isfinite(reference).all(): raise ValueError('Nonfinite ACT values')
    base_q,base_steps=quantize_all(device)
    selected,scores=choose_group(device,base_q,base_steps)
    candidates,order=create_candidates(device,base_q,base_steps,selected)
    with image.open('rb') as fd, mmap.mmap(fd.fileno(),0,access=mmap.ACCESS_READ) as mm:
        ref,device_exact,outputs=scan(mm,layout,device,reference,candidates,selected,base_q,base_steps)
    baseline=metric(outputs['baseline'],ref)
    upstream=metric(device_exact,ref)
    rows=[]
    for c in candidates:
        stats=metric(outputs[c['name']],ref)
        reduction=100*(baseline['mixed_failures']-stats['mixed_failures'])/baseline['mixed_failures'] if baseline['mixed_failures'] else 0.0
        qualifies=c['kind']!='baseline' and stats['mixed_failures']<=10 and reduction>=80
        rows.append({'name':c['name'],'kind':c['kind'],'peeled_positions':c['indices'].tolist(),
                     'quantization_step':float(c['step']),'statistics':stats,'failure_reduction_percent':reduction,
                     'offline_precision_gate':qualifies})
    print('PHASE60 | ONE-GROUP OUTLIER-AWARE Q15 PRECISION GATE | OFFLINE ONLY')
    print('No firmware, no COM3, no real-time PC model inference. Original pretrained INT4 image.')
    print(f'SELECTED GROUP {selected}/37 | original step={base_steps[selected]:.9g} | top local device positions={order[:6].tolist()}')
    print(f'UPSTREAM input-only failures={upstream["mixed_failures"]}/896')
    print('variant       | Q15 step     | sparse taps | mixed failures | max abs      | reduction | gate')
    for r in rows:
        print(f'{r["name"]:<13} | {r["quantization_step"]:>12.6g} | {len(r["peeled_positions"]):>11} | '
              f'{r["statistics"]["mixed_failures"]:>4}/896       | {r["statistics"]["max_abs"]:>12.8g} | '
              f'{r["failure_reduction_percent"]:>7.2f}% | {"PASS" if r["offline_precision_gate"] else "REJECT"}')
    finalists=[r for r in rows if r['offline_precision_gate']]
    sanity = baseline['mixed_failures']==482 and selected==6
    if not sanity:
        print(f'BASELINE REPRODUCTION WARNING: expected group 6 and 482 baseline failures; got group {selected}, failures {baseline["mixed_failures"]}. Do NOT adopt candidates without investigating.')
    elif finalists:
        best=min(finalists,key=lambda r:(len(r['peeled_positions']),r['statistics']['mixed_failures']))
        print(f'PHASE60 OFFLINE GATE PASS: {best["name"]} with {best["statistics"]["mixed_failures"]} failures; requires hardware speed and paired accuracy tests.')
    else:
        print('PHASE60 OFFLINE GATE FAIL: no acceptable candidate. Leave firmware unchanged.')
    report={'model_sha256_expected':EXPECTED_SHA,'baseline_reproduced':sanity,'selected_group':selected,
            'selected_score':float(scores[selected]),'upstream':upstream,'baseline':baseline,'variants':rows,
            'warning':'Precision replay only; sparse scalar read overhead and runtime not measured on ESP32; full 24-layer accuracy unknown.'}
    Path(args.report).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('REPORT',args.report)
    return report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--trace',default='phase49_act_trace.npz')
    ap.add_argument('--metadata',default='phase25_qwen_24layers_int4.json')
    ap.add_argument('--image',default='phase25_qwen_24layers_int4.bin')
    ap.add_argument('--report',default='phase60_outlier_quantization_report.json')
    ap.add_argument('--verify-sha',action='store_true')
    run(ap.parse_args())


if __name__=='__main__': main()
