"""Phase 66: offline attribution of layer-4 MLP projection errors.

Replays real pretrained INT4 matrices from the user's local model image.
No COM3, firmware changes, or PC computations during ESP32 inference.
Device and reference tensors come from a completed Phase66 capture.
"""
import argparse
import hashlib
import json
import mmap
from pathlib import Path
import numpy as np

EXPECTED_SHA = '9cb8e2b5e08ab6594f32c3dfd255c96e4ba52e7d6b02586c65cfc928d019ce43'
GROUP = 128
ATOL = .001
RTOL = .001


def quant(v):
    v=np.asarray(v,np.float32)
    n=float(np.max(np.abs(v)))
    if not n: return np.zeros(len(v),np.int16),np.float32(1)
    scale=np.float32(np.float32(n)/np.float32(32760))
    inv=np.float32(np.float32(32760)/np.float32(n))
    q=np.clip(np.rint(v*inv),-32760,32760).astype(np.int16)
    return q,scale


def metrics(candidate, reference):
    delta=np.abs(candidate.astype(np.float64)-reference.astype(np.float64))
    tol=ATOL+RTOL*np.abs(reference.astype(np.float64))
    return {'failures':int(np.count_nonzero(delta>tol)), 'count':int(len(delta)),
            'max_abs':float(delta.max()),'mean_abs':float(delta.mean())}


def replay(mm, matrix, device_x, ref_x, peel_group=None, peel_pos=None):
    cols=int(matrix['cols']); rows=int(matrix['rows']); rb=int(matrix['row_bytes']); offset=int(matrix['offset'])
    if cols % GROUP or len(device_x)!=cols or len(ref_x)!=cols or rb<cols//2+4:
        raise ValueError('Unexpected INT4 matrix layout or input dimension')
    num_groups=cols//GROUP
    q=[]; steps=[]
    for i in range(num_groups):
        z,s=quant(device_x[i*GROUP:(i+1)*GROUP]);q.append(z);steps.append(s)
    if peel_group is not None:
        v=device_x[peel_group*GROUP:(peel_group+1)*GROUP]
        if peel_pos is None: peel_pos=int(np.argmax(np.abs(v)))
        vv=v.copy();vv[peel_pos]=0
        q_new, step_new =quant(vv)
    ref_out=np.empty(rows,np.float32); exact_device=np.empty(rows,np.float32)
    q15_out=np.empty(rows,np.float32)
    peeled_out=np.empty(rows,np.float32) if peel_group is not None else None
    for start in range(0,rows,16):
        end=min(start+16,rows)
        raw=np.frombuffer(mm[offset+start*rb:offset+end*rb],dtype=np.uint8)
        if raw.size!=(end-start)*rb: raise RuntimeError('Missing INT4 matrix row bytes')
        packed=raw.reshape(end-start,rb)
        b=packed[:,:cols//2]
        low=(b&15).astype(np.int8);high=(b>>4).astype(np.int8)
        low[low>=8]-=16;high[high>=8]-=16
        w=np.empty((end-start,cols),dtype=np.int8)
        w[:,::2]=low;w[:,1::2]=high
        scales=np.frombuffer(packed[:,cols//2:cols//2+4].copy().tobytes(),dtype='<f4')
        # Exact INT4 weight products with FP64 accumulation: offline diagnostic only.
        f64=w.astype(np.float64)
        ref_out[start:end]=((f64@ref_x.astype(np.float64))*scales).astype(np.float32)
        exact_device[start:end]=((f64@device_x.astype(np.float64))*scales).astype(np.float32)
        base=np.zeros(end-start,dtype=np.float32)
        for g in range(num_groups):
            dot=(w[:,g*GROUP:(g+1)*GROUP].astype(np.int32)@q[g].astype(np.int32)).astype(np.float32)
            base+=dot*steps[g]
        q15_out[start:end]=base*scales
        if peeled_out is not None:
            g=peel_group; segment=w[:,g*GROUP:(g+1)*GROUP].astype(np.int32)
            original=(segment@q[g].astype(np.int32)).astype(np.float32)*steps[g]
            substitute=(segment@q_new.astype(np.int32)).astype(np.float32)*step_new
            substitute+=w[:,g*GROUP+peel_pos].astype(np.float32)*np.float32(device_x[g*GROUP+peel_pos])
            peeled_out[start:end]=(base-original+substitute)*scales
    return ref_out,exact_device,q15_out,peeled_out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trace',default='phase66_layer4_trace.npz')
    p.add_argument('--metadata',default='phase25_qwen_24layers_int4.json')
    p.add_argument('--image',default='phase25_qwen_24layers_int4.bin')
    p.add_argument('--report',default='phase66_attribution_report.json')
    p.add_argument('--verify-sha',action='store_true')
    a=p.parse_args()
    with np.load(a.trace,allow_pickle=False) as z:
        vals={k:np.asarray(z[k],np.float32).copy() for k in z.files}
    for name,n in [('NORM2',896),('GATE',4864),('UP',4864),('ACT',4864),('MLP_OUT',896)]:
        for label in ('device_','reference_'):
            key=label+name
            if key not in vals or vals[key].shape!=(n,) or not np.all(np.isfinite(vals[key])):
                raise ValueError(f'Invalid or absent trace vector: {key}')
    meta=json.loads(Path(a.metadata).read_text(encoding='utf8'))
    if Path(a.image).stat().st_size!=int(meta['image_size']):raise ValueError('Model image size mismatch')
    if a.verify_sha:
        with open(a.image,'rb') as f:
            if hashlib.file_digest(f,'sha256').hexdigest()!=EXPECTED_SHA:raise ValueError('Model SHA mismatch')
    mats=meta['layers'][3]['matrices']
    print('PHASE66 | LAYER4 MLP ERROR ATTRIBUTION | OFFLINE ONLY',flush=True)
    print('Reference computed after ESP32 capture. Genuine pretrained INT4 weights.',flush=True)
    records={}
    with open(a.image,'rb') as fd, mmap.mmap(fd.fileno(),0,access=mmap.ACCESS_READ) as mm:
        for name,mi,xname in [('GATE',4,'NORM2'),('UP',5,'NORM2'),('DOWN',6,'ACT')]:
            dx=vals['device_'+xname];rx=vals['reference_'+xname]
            group=None;pos=None
            if name=='DOWN':
                score=[]
                for g in range(len(dx)//GROUP):
                    q,step=quant(dx[g*GROUP:(g+1)*GROUP])
                    residual=dx[g*GROUP:(g+1)*GROUP]-q.astype(np.float32)*step
                    score.append(float(np.abs(residual.astype(np.float64)).sum()))
                group=int(np.argmax(score))
                pos=int(np.argmax(np.abs(dx[group*GROUP:(group+1)*GROUP])))
            reference,upstream,base,peeled=replay(mm,mats[mi],dx,rx,group,pos)
            r={'input_exact_vs_reference':metrics(upstream,reference),
               'grouped_q15_vs_reference':metrics(base,reference),
               'q15_vs_device_exact':metrics(base,upstream)}
            if name=='DOWN':
                r['selected_group']=group;r['outlier_local_index']=pos
                r['peel_top1_vs_reference']=metrics(peeled,reference)
                r['peel_top1_vs_device_exact']=metrics(peeled,upstream)
                r['actual_esp32_vs_reference_trace']=metrics(vals['device_MLP_OUT'],vals['reference_MLP_OUT'])
                r['replay_q15_vs_esp32_trace']=metrics(base,vals['device_MLP_OUT'])
            else:
                r['actual_esp32_vs_reference_trace']=metrics(vals['device_'+name],vals['reference_'+name])
                r['replay_q15_vs_esp32_trace']=metrics(base,vals['device_'+name])
            records[name]=r
            print(f'\n{name} ({int(mats[mi]["rows"])}x{int(mats[mi]["cols"])})',flush=True)
            for label,score in r.items():
                if isinstance(score,dict):
                    print(f'  {label:35} failures={score["failures"]:4}/{score["count"]} max_abs={score["max_abs"]:.9g}',flush=True)
            if name=='DOWN':
                print('  peel candidate device-only group=',group,'outlier=',pos,flush=True)
    # Important: these are isolated projections, not a full layer with corrections applied.
    summary='OFFLINE ATTRIBUTION ONLY: no firmware changes or validated layer-4 fix.'
    print('\nPHASE66:',summary,flush=True)
    Path(a.report).write_text(json.dumps({'results':records,'warning':summary},indent=2),encoding='utf8')
    print('REPORT',a.report,flush=True)

if __name__=='__main__':main()
