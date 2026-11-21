"""Phase 53 offline audit: attribute layer-3 down-projection errors from a saved ESP32 trace.

No COM3, no firmware changes, and no PC-assisted token generation.
Uses original INT4 matrix and saved device/reference ACT vectors, then replays
ESP32 grouped-Q15 (128-wide) activation quantization after hardware inference.
"""
import argparse
import hashlib
import json
import mmap
from pathlib import Path
import numpy as np


def matrix_layout(metadata):
    m = metadata['layers'][2]['matrices'][6]
    cols, rows, row_bytes = (int(m[x]) for x in ('cols', 'rows', 'row_bytes'))
    if (rows, cols) != (896, 4864) or row_bytes < cols//2 + 4:
        raise ValueError('Unexpected layer-3 down-projection geometry')
    return int(m['offset']), rows, cols, row_bytes


def grouped_q15(x, group_size=128):
    """Replicate C quantize_q15: per-group float32 scale and nearest-even rounding."""
    x=np.asarray(x,dtype=np.float32)
    if x.shape != (4864,): raise ValueError('ACT shape must be (4864,)')
    q=np.empty(x.shape,dtype=np.int16)
    steps=np.empty((len(x)+group_size-1)//group_size,dtype=np.float32)
    for g, i in enumerate(range(0,len(x),group_size)):
        z=x[i:i+group_size]
        maxabs=np.max(np.abs(z))
        if maxabs == 0:
            q[i:i+group_size]=0
            steps[g]=np.float32(1)
        else:
            inv=np.float32(np.float32(32760.0)/maxabs)
            steps[g]=np.float32(maxabs/np.float32(32760.0))
            q[i:i+group_size]=np.clip(np.rint(z*inv),-32760,32760).astype(np.int16)
    return q,steps


def projections(mm, layout, device, reference, row_batch=16):
    offset,rows,cols,row_bytes=layout
    q,steps=grouped_q15(device)
    out=np.empty((3,rows), dtype=np.float32)
    for start in range(0,rows,row_batch):
        count=min(row_batch,rows-start)
        raw=np.frombuffer(mm[offset+start*row_bytes:offset+(start+count)*row_bytes],dtype=np.uint8)
        chunk=raw.reshape(count,row_bytes)
        packed=chunk[:,:cols//2]
        low=(packed & 15).astype(np.int8)
        high=(packed >> 4).astype(np.int8)
        low[low >= 8] -= 16
        high[high >= 8] -= 16
        w=np.empty((count,cols),dtype=np.float32)
        w[:,0::2]=low
        w[:,1::2]=high
        scales=np.frombuffer(chunk[:,cols//2:cols//2+4].copy().tobytes(), dtype='<f4')
        # double precision accumulation isolates the mathematical contributions;
        # the grouped-Q15 sum is separately accumulated in float32 like device C.
        exact_dev=(w.astype(np.float64) @ device.astype(np.float64)).astype(np.float32)*scales
        exact_ref=(w.astype(np.float64) @ reference.astype(np.float64)).astype(np.float32)*scales
        approx=np.zeros(count,dtype=np.float32)
        for group, i in enumerate(range(0,cols,128)):
            # Every group's maximum |dot| <= 128*8*32760, safely within int32.
            dots=w[:,i:i+128].astype(np.int32) @ q[i:i+128].astype(np.int32)
            approx += dots.astype(np.float32)*steps[group]
        approx *= scales
        out[0,start:start+count]=exact_dev
        out[1,start:start+count]=exact_ref
        out[2,start:start+count]=approx
    return out


def metrics(v, ref=None):
    a=np.abs(np.asarray(v,dtype=np.float64))
    d={'max_abs':float(a.max()),'mean_abs':float(a.mean()),'rms':float(np.sqrt(np.mean(a*a))),
       'count_above_0_001':int(np.count_nonzero(a>0.001)),
       'count_above_0_01':int(np.count_nonzero(a>0.01))}
    if ref is not None:
        r=np.asarray(ref,dtype=np.float64)
        allow=0.001+0.001*np.abs(r)
        d['combined_tolerance_failures']=int(np.count_nonzero(a>allow))
    return d


def analyze(trace,mm,layout):
    x=np.asarray(trace['device_ACT'],dtype=np.float32)
    y=np.asarray(trace['reference_ACT'],dtype=np.float32)
    if x.shape != (4864,) or y.shape != (4864,) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Expected finite 4864-value device/reference ACT vectors')
    fp_device,fp_reference,grouped_device=projections(mm,layout,x,y)
    upstream=fp_device-fp_reference
    quant=grouped_device-fp_device
    net=grouped_device-fp_reference
    recon=upstream+quant
    result={'upstream_ACT_error_only':metrics(upstream,fp_reference),
            'grouped_Q15_projection_error_only':metrics(quant,fp_device),
            'combined_projected_error':metrics(net,fp_reference),
            'reconstruction_max_difference':float(np.max(np.abs(net-recon))),
            'observed_phase48_MLP_OUT_max_error':0.0100708008,
            'observed_phase48_MLP_OUT_abs_failures':605,
            'full_device_MLP_OUT_vector_available':False,
            'interpretation_limit':'Replay is approximate due to float rounding and lacks a matching captured device MLP_OUT vector. It cannot prove the exact cause of the historical 605 output failures.'}
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--trace',default='phase49_act_trace.npz')
    ap.add_argument('--metadata',default='phase25_qwen_24layers_int4.json')
    ap.add_argument('--image',default='phase25_qwen_24layers_int4.bin')
    ap.add_argument('--report',default='phase53_downprojection_audit.json')
    ap.add_argument('--verify-sha',action='store_true')
    args=ap.parse_args()
    meta=json.loads(Path(args.metadata).read_text(encoding='utf-8'))
    file=Path(args.image)
    if file.stat().st_size != int(meta['image_size']): raise ValueError('Model size mismatch')
    if args.verify_sha:
        with file.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
        if digest != meta['sha256']: raise ValueError('Model SHA mismatch')
    with np.load(args.trace,allow_pickle=False) as trace, file.open('rb') as fp:
        with mmap.mmap(fp.fileno(),0,access=mmap.ACCESS_READ) as mm:
            result=analyze(trace,mm,matrix_layout(meta))
    print('PHASE53 | OFFLINE LAYER-3 DOWN-PROJECTION AUDIT')
    print('No COM3, no flashing, no host-assisted generation; original INT4 pretrained weights.')
    for key in ('upstream_ACT_error_only','grouped_Q15_projection_error_only','combined_projected_error'):
        v=result[key]
        print(f'{key}: max={v["max_abs"]:.9g}, mean={v["mean_abs"]:.9g}, rms={v["rms"]:.9g}, '
              f'>0.001={v["count_above_0_001"]}/896, mixed_tol_failures={v["combined_tolerance_failures"]}/896')
    print('ADDITIVITY CHECK:',result['reconstruction_max_difference'])
    print('HISTORICAL DEVICE MLP_OUT: max=0.0100708008, >0.001=605/896 (not paired in saved trace)')
    print('CAUTION: Neither offline replay nor its tolerance proves full-model correctness.')
    Path(args.report).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('REPORT:',args.report)

if __name__=='__main__': main()
