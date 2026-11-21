"""Phase 54: offline layer-3 grouped-Q15 precision vs overhead tradeoff.

Uses recorded ESP32 and NumPy ACT vectors from Phase 49 and *original* pretrained
INT4 layer-3 down weights. All calculations are post-hardware diagnostics,
NOT host-assisted inference. Nothing here updates ESP32 firmware or weights.
"""
import argparse
import hashlib
import json
import mmap
from pathlib import Path

import numpy as np

from phase53_downprojection_audit import matrix_layout, grouped_q15, metrics

COLS = 4864
ROWS = 896


def scan(mm, layout, device, reference, candidates=(128, 64, 32, 16), row_batch=16):
    offset, rows, cols, row_bytes = layout
    if (rows, cols) != (ROWS, COLS):
        raise ValueError('Unexpected down-projection geometry')
    vectors = {}
    for size in candidates:
        if size <= 0 or cols % size:
            raise ValueError(f'Invalid group size {size}; must divide 4864')
        vectors[size] = (*grouped_q15(device, size), np.empty(rows, np.float32))
    ref_out = np.empty(rows, np.float32)
    device_exact_out = np.empty(rows, np.float32)
    for start in range(0, rows, row_batch):
        count = min(row_batch, rows-start)
        chunk = np.frombuffer(mm[offset+start*row_bytes: offset+(start+count)*row_bytes], dtype=np.uint8).reshape(count, row_bytes)
        packed=chunk[:, :cols//2]
        low=(packed & 15).astype(np.int8)
        high=(packed >> 4).astype(np.int8)
        low[low>=8]-=16
        high[high>=8]-=16
        w=np.empty((count, cols), np.int8)
        w[:, ::2]=low
        w[:, 1::2]=high
        scales=np.frombuffer(chunk[:,cols//2:cols//2+4].copy().tobytes(),dtype='<f4')
        w64 = w.astype(np.float64)
        ref_out[start:start+count]=(w64 @ reference.astype(np.float64)).astype(np.float32)*scales
        device_exact_out[start:start+count]=(w64 @ device.astype(np.float64)).astype(np.float32)*scales
        for size, (q, steps, output) in vectors.items():
            # Identical group-accumulation pattern to existing grouped Q15 replay.
            acc = np.zeros(count, np.float32)
            for group, i in enumerate(range(0,cols,size)):
                dots = w[:,i:i+size].astype(np.int32) @ q[i:i+size].astype(np.int32)
                acc += dots.astype(np.float32) * steps[group]
            output[start:start+count]=acc*scales
    return ref_out, device_exact_out, {k: v[2] for k,v in vectors.items()}


def summarize(ref, exact, approximations, max_failures=10, min_reduction=0.8):
    upstream=metrics(exact-ref, ref)
    baseline=None
    rows=[]
    for size in sorted(approximations, reverse=True):
        pred=approximations[size]
        qerr=metrics(pred-exact, exact)
        combined=metrics(pred-ref, ref)
        if size==128: baseline=combined['combined_tolerance_failures']
        rows.append({'group_size':size, 'group_count':COLS//size,
                     'relative_group_work_vs_128':(COLS//size)/(COLS//128),
                     'q15_projection_error':qerr,
                     'combined_error':combined})
    if baseline is None:
        raise ValueError('128-feature baseline missing')
    for r in rows:
        failures=r['combined_error']['combined_tolerance_failures']
        r['failure_reduction_vs_128']=(baseline-failures)/baseline if baseline else (1 if failures==0 else 0)
        r['offline_precision_gate']=(r['group_size']!=128 and failures<=max_failures and
                                     r['failure_reduction_vs_128']>=min_reduction)
    return {'upstream_error':upstream,'baseline_128_failures':baseline,'candidates':rows,
            'interpretation':'Offline error counterfactual only. Group count is NOT an ESP32 speed benchmark. Do not promote a candidate until a short ESP32 kernel microbenchmark proves it is not materially slower and a paired hardware output validates accuracy.'}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--trace', default='phase49_act_trace.npz')
    ap.add_argument('--metadata', default='phase25_qwen_24layers_int4.json')
    ap.add_argument('--image', default='phase25_qwen_24layers_int4.bin')
    ap.add_argument('--report', default='phase54_group_precision_report.json')
    ap.add_argument('--verify-sha',action='store_true')
    a=ap.parse_args()
    meta=json.loads(Path(a.metadata).read_text(encoding='utf-8'))
    file=Path(a.image)
    if file.stat().st_size!=int(meta['image_size']): raise ValueError('Model image size mismatch')
    if a.verify_sha:
        with file.open('rb') as f:
            if hashlib.file_digest(f,'sha256').hexdigest()!=meta['sha256']:
                raise ValueError('Model hash mismatch')
    with np.load(a.trace,allow_pickle=False) as t:
        device=np.asarray(t['device_ACT'],dtype=np.float32).copy()
        reference=np.asarray(t['reference_ACT'],dtype=np.float32).copy()
    if device.shape!=(COLS,) or reference.shape!=(COLS,):
        raise ValueError('Trace missing 4864-feature ACT vectors')
    if not np.isfinite(device).all() or not np.isfinite(reference).all():
        raise ValueError('Nonfinite trace values')
    with file.open('rb') as f, mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as mm:
        ref,exact,preds=scan(mm,matrix_layout(meta),device,reference)
    report=summarize(ref,exact,preds)
    print('PHASE54 | GENUINE INT4 GROUPED-Q15 PRECISION GATE | OFFLINE ONLY')
    print('USB Type-C, model image, firmware and inference kernel unchanged.')
    print('Source: previously recorded Phase49 ACT + genuine pretrained layer-3 INT4 down weights.')
    print('Pre-existing ACT input error: mixed failures',report['upstream_error']['combined_tolerance_failures'])
    print('group | groups | group-work proxy | projection max abs | combined mixed fails | reduction vs128 | candidate')
    for r in report['candidates']:
        m=r['combined_error']; q=r['q15_projection_error']
        print(f"{r['group_size']:>5} | {r['group_count']:>6} | {r['relative_group_work_vs_128']:>5.1f}x | "
              f"{q['max_abs']:.7g} | {m['combined_tolerance_failures']:>4}/896 | "
              f"{r['failure_reduction_vs_128']*100:>6.1f}% | {'INVESTIGATE' if r['offline_precision_gate'] else 'REJECT/BASELINE'}")
    finalists=[r for r in report['candidates'] if r['offline_precision_gate']]
    if finalists:
        best=max(finalists,key=lambda x:x['group_size'])
        print(f"OFFLINE PRECISION CANDIDATE: group={best['group_size']}. NOT ADOPTED. Must pass short ESP32 speed/accuracy gate first.")
    else:
        print('PHASE54 OFFLINE GATE FAIL: no candidate meets >=80% failure reduction and <=10 mixed failures. Do not modify firmware.')
    Path(a.report).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('REPORT', a.report)

if __name__=='__main__': main()
