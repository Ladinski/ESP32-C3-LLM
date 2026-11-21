"""Phase 55: post-inference, offline, selective INT4 down-projection precision experiment.

Uses real ESP32 ACT trace and original checkpoint INT4 row bytes. No serial port,
firmware changes, prompt inference, or PC-driven token selection. The PC replays
counterfactual arithmetic after the ESP32 inference already took place.
"""
import argparse
import hashlib
import json
import mmap
from pathlib import Path
import numpy as np

COLS=4864
ROWS=896
GROUP=128
GROUPS=COLS//GROUP
ATOL=0.001
RTOL=0.001


def quantize(x):
    q=np.empty((GROUPS,GROUP),dtype=np.int16)
    steps=np.empty(GROUPS,dtype=np.float32)
    residuals=np.empty((GROUPS,GROUP),dtype=np.float32)
    for i in range(GROUPS):
        z=x[i*GROUP:(i+1)*GROUP]
        m=float(np.max(np.abs(z)))
        if m==0:
            q[i]=0
            steps[i]=1
        else:
            inv=np.float32(np.float32(32760)/np.float32(m))
            steps[i]=np.float32(np.float32(m)/np.float32(32760))
            q[i]=np.clip(np.rint(z*inv),-32760,32760).astype(np.int16)
        residuals[i]=z-q[i].astype(np.float32)*steps[i]
    return q,steps,residuals


def rank_groups(x,residuals,method):
    v=np.abs(residuals.astype(np.float64))
    if method=='residual_l1': score=np.sum(v,axis=1)
    elif method=='residual_rms': score=np.sqrt(np.mean(v*v,axis=1))
    elif method=='residual_max': score=np.max(v,axis=1)
    elif method=='activation_max': score=np.max(np.abs(x.reshape(GROUPS,GROUP)),axis=1)
    else: raise ValueError(method)
    return np.argsort(-score,kind='stable'),score


def read_row_chunk(mm, offset, start, end, row_bytes):
    """Read [start, end) rows using absolute image offsets."""
    first = offset + start * row_bytes
    last = offset + end * row_bytes
    raw = np.frombuffer(mm[first:last], dtype=np.uint8)
    expected = (end - start) * row_bytes
    if raw.size != expected:
        raise ValueError(f"Incomplete INT4 row data at rows {start}:{end}: "
                         f"expected {expected} bytes, got {raw.size}")
    return raw.reshape(end - start, row_bytes)


def evaluate(mm,layout,x,ref,counts,selectors):
    offset,rows,cols,row_bytes=layout
    assert (rows,cols)==(ROWS,COLS)
    q,steps,residuals=quantize(x)
    ranking={method:rank_groups(x,residuals,method)[0] for method in selectors}
    scores={method:rank_groups(x,residuals,method)[1] for method in selectors}
    # Collect all 38 group correction components for real row-wise pretrained weights.
    # A hypothetical corrected group computes its FP32 result on ESP32, not on PC.
    correction=np.empty((GROUPS,ROWS),dtype=np.float32)
    base=np.empty(ROWS,dtype=np.float32)
    exact_ref=np.empty(ROWS,dtype=np.float32)
    exact_dev=np.empty(ROWS,dtype=np.float32)
    for start in range(0,ROWS,16):
        end=min(start+16,ROWS)
        lines=read_row_chunk(mm,offset,start,end,row_bytes)
        packed=lines[:,:cols//2]
        low=(packed & 15).astype(np.int8)
        high=(packed >> 4).astype(np.int8)
        low[low>=8]-=16
        high[high>=8]-=16
        w=np.empty((end-start,COLS),dtype=np.int8)
        w[:,::2]=low
        w[:,1::2]=high
        scales=np.frombuffer(lines[:,cols//2:cols//2+4].copy().tobytes(),dtype='<f4')
        exact_ref[start:end]=((w.astype(np.float64) @ ref.astype(np.float64)).astype(np.float32))*scales
        exact_dev[start:end]=((w.astype(np.float64) @ x.astype(np.float64)).astype(np.float32))*scales
        acc=np.zeros(end-start,dtype=np.float32)
        for g in range(GROUPS):
            i=g*GROUP
            wg=w[:,i:i+GROUP]
            ints=wg.astype(np.int32)@q[g].astype(np.int32)
            approximate=ints.astype(np.float32)*steps[g]
            acc += approximate
            precise=(wg.astype(np.float64)@x[i:i+GROUP].astype(np.float64)).astype(np.float32)
            correction[g,start:end]=(precise-approximate)*scales
        base[start:end]=acc*scales
    output={}
    for method in selectors:
        ordered=ranking[method]
        variants=[]
        for k in counts:
            selected=ordered[:k]
            predicted=base.copy()
            if k:
                predicted+=np.sum(correction[selected],axis=0,dtype=np.float32)
            delta=np.abs(predicted-exact_ref)
            allowance=ATOL+RTOL*np.abs(exact_ref)
            group_fraction=k/GROUPS
            variants.append({'corrected_groups':int(k),
                             'share_of_groups_corrected':float(group_fraction),
                             'max_absolute_error':float(np.max(delta)),
                             'mean_absolute_error':float(np.mean(delta)),
                             'mixed_tolerance_failures':int(np.count_nonzero(delta>allowance)),
                             'max_normalized_error':float(np.max(delta/np.maximum(np.abs(exact_ref),1))),
                             'requires_fp32_group_mac_fraction':float(group_fraction),
                             'estimated_additional_dot_work_not_runtime':float(group_fraction)})
        output[method]={'top_group_indices':ordered[:min(8,GROUPS)].tolist(),
                        'top_group_scores':[float(scores[method][g]) for g in ordered[:min(8,GROUPS)]],
                        'variants':variants}
    upstream=np.abs(exact_dev-exact_ref)
    upstream_fail=int(np.count_nonzero(upstream>ATOL+RTOL*np.abs(exact_ref)))
    return {'baseline_tolerance':{'atol':ATOL,'rtol':RTOL},
            'input_only_failures':upstream_fail,
            'selectors':output,
            'note':'Oracle-free group selection uses only the ESP32 ACT vector. FP32 group correction values are hypothetical; on-device timing, memory use, and paired output verification are NOT measured. The replay does not modify any model weight, or use the PC to generate a token.'}


def matrix_layout(meta):
    m=meta['layers'][2]['matrices'][6]
    offset,rows,cols,row_bytes=[int(m[k]) for k in ('offset','rows','cols','row_bytes')]
    if (rows,cols)!=(ROWS,COLS) or row_bytes<cols//2+4:
        raise ValueError('Unexpected pretrained layer-3 down-projection layout')
    return offset,rows,cols,row_bytes


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--trace',default='phase49_act_trace.npz')
    ap.add_argument('--metadata',default='phase25_qwen_24layers_int4.json')
    ap.add_argument('--image',default='phase25_qwen_24layers_int4.bin')
    ap.add_argument('--report',default='phase55_selective_report.json')
    ap.add_argument('--verify-sha',action='store_true')
    args=ap.parse_args()
    meta=json.loads(Path(args.metadata).read_text(encoding='utf-8'))
    image=Path(args.image)
    if image.stat().st_size!=int(meta['image_size']): raise ValueError('Model image size mismatch')
    if args.verify_sha:
        with image.open('rb') as fp:
            if hashlib.file_digest(fp,'sha256').hexdigest()!=meta['sha256']:
                raise ValueError('Model image SHA mismatch')
    with np.load(args.trace,allow_pickle=False) as trace:
        x=np.array(trace['device_ACT'],dtype=np.float32)
        ref=np.array(trace['reference_ACT'],dtype=np.float32)
    if x.shape!=(COLS,) or ref.shape!=(COLS,): raise ValueError('Expected 4864-element ACT vectors')
    if not (np.isfinite(x).all() and np.isfinite(ref).all()):raise ValueError('Nonfinite ACT values')
    groups=(0,1,2,4,8,12,16,24,38)
    selectors=('residual_l1','residual_rms','residual_max','activation_max')
    with image.open('rb') as fp,mmap.mmap(fp.fileno(),0,access=mmap.ACCESS_READ) as mm:
        results=evaluate(mm,matrix_layout(meta),x,ref,groups,selectors)
    print('PHASE55 | SELECTIVE PRECISION CORRECTION GATE | OFFLINE ONLY')
    print('No COM3 or firmware change. Original pretrained INT4 down-projection.')
    print('Selectors use DEVICE activations only, never reference deltas.')
    print('Upstream-only combined tolerance failures:',results['input_only_failures'])
    for method,data in results['selectors'].items():
        print('\nSELECTOR',method,'top groups',data['top_group_indices'])
        print('  corrected | error max | mixed failures /896 | FP32 group fraction (NOT measured speed)')
        for r in data['variants']:
            print(f"  {r['corrected_groups']:>9} | {r['max_absolute_error']:.8g} | {r['mixed_tolerance_failures']:>4}/896 | {r['share_of_groups_corrected']*100:5.1f}%")
    viable=[]
    for method,data in results['selectors'].items():
        for r in data['variants']:
            if r['mixed_tolerance_failures']<=10 and r['corrected_groups']<GROUPS:
                viable.append((r['corrected_groups'],method,r))
    if not viable:
        print('\nPHASE55 GATE FAIL: no selective candidate <=10 combined failures. DO NOT change firmware.')
    else:
        k,method,record=min(viable,key=lambda v:(v[0],v[2]['mixed_tolerance_failures']))
        print(f'\nPHASE55 OFFLINE CANDIDATE: {method} with {k}/{GROUPS} groups corrected; {record["mixed_tolerance_failures"]} failures.')
        print('NOT ACCEPTED FOR ESP32: on-device benchmark must demonstrate no meaningful slowdown, and paired hardware accuracy must pass.')
    Path(args.report).write_text(json.dumps(results,indent=2),encoding='utf-8')
    print('REPORT:',args.report)

if __name__=='__main__':main()
