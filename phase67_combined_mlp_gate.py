"""Offline combined layer-4 MLP correction search; ESP32 trace and pretrained INT4 only.

All selection uses the device-side input, never reference residuals.
The PC does NOT participate in inference: this script runs after the completed capture.
"""
import argparse, hashlib, itertools, json, mmap
from pathlib import Path
import numpy as np

SHA='9cb8e2b5e08ab6594f32c3dfd255c96e4ba52e7d6b02586c65cfc928d019ce43'
GROUP=128
ATOL=RTOL=.001


def quant(v):
    v=np.asarray(v,np.float32)
    maximum=np.float32(np.max(np.abs(v)))
    if maximum==0:return np.zeros(v.shape,np.int16),np.float32(1.)
    step=np.float32(maximum/np.float32(32760))
    inv=np.float32(np.float32(32760)/maximum)
    return np.clip(np.rint(v*inv),-32760,32760).astype(np.int16),step


def metrics(a,b):
    d=np.abs(np.asarray(a,np.float64)-np.asarray(b,np.float64))
    tol=ATOL+RTOL*np.abs(np.asarray(b,np.float64))
    return {'failures':int(np.count_nonzero(d>tol)),'count':int(d.size),
            'max_abs':float(np.max(d)),'mean_abs':float(np.mean(d))}


def plan(x,k):
    x=np.asarray(x,np.float32)
    groups=x.size//GROUP
    qs=[];steps=[];scores=[]
    for g in range(groups):
        s=x[g*GROUP:(g+1)*GROUP]
        q,step=quant(s)
        qs.append(q);steps.append(step)
        scores.append(float(np.abs(s.astype(np.float64)-q.astype(np.float64)*step).sum()))
    g=int(np.argmax(scores))
    order=np.argsort(-np.abs(x[g*GROUP:(g+1)*GROUP]),kind='stable')
    selected=order[:k].astype(np.int64)
    section=x[g*GROUP:(g+1)*GROUP].copy();section[selected]=0
    pq,ps=quant(section)
    return {'group':g,'indices':selected,'values':x[g*GROUP+selected].copy(),
            'q':pq,'step':ps,'qs':qs,'steps':steps,'score':scores[g]}


def decode_matrix(mm,layout):
    rows=int(layout['rows']);cols=int(layout['cols']);rb=int(layout['row_bytes']);offset=int(layout['offset'])
    if cols%GROUP or rb<cols//2+4 or offset<0 or offset+rows*rb>len(mm):
        raise ValueError('Invalid INT4 matrix layout')
    weights=np.empty((rows,cols),np.int8)
    scales=np.empty(rows,np.float32)
    for start in range(0,rows,32):
        end=min(start+32,rows)
        raw=np.frombuffer(mm[offset+start*rb:offset+end*rb],np.uint8)
        if raw.size!=(end-start)*rb:raise ValueError('Truncated INT4 matrix')
        lines=raw.reshape(end-start,rb)
        b=lines[:,:cols//2]
        lo=(b&15).astype(np.int8);hi=(b>>4).astype(np.int8)
        lo[lo>=8]-=16;hi[hi>=8]-=16
        weights[start:end,::2]=lo;weights[start:end,1::2]=hi
        scales[start:end]=np.frombuffer(lines[:,cols//2:cols//2+4].copy().tobytes(),dtype='<f4')
    return weights,scales


def project(weight,scales,x,k=0):
    x=np.asarray(x,np.float32)
    rows,cols=weight.shape
    if x.shape!=(cols,):raise ValueError('Wrong vector shape')
    p=plan(x,k) if k else None
    accum=np.zeros(rows,np.float32)
    for g in range(cols//GROUP):
        v=x[g*GROUP:(g+1)*GROUP]
        if p is not None and g==p['group']:
            q,step=p['q'],p['step']
        else:q,step=quant(v)
        # int32 safe: max 128 * 32760 * 8 < 2**31
        dots=weight[:,g*GROUP:(g+1)*GROUP].astype(np.int32)@q.astype(np.int32)
        accum+=dots.astype(np.float32)*step
        if p is not None and g==p['group']:
            for i,val in zip(p['indices'],p['values']):
                accum+=weight[:,g*GROUP+int(i)].astype(np.float32)*val
    return accum*scales,p


def swiglu(g,u):
    # same algebra as firmware; use float32 intermediates
    return np.asarray((g/(np.float32(1)+np.exp(-g).astype(np.float32)))*u,np.float32)


def run():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trace',default='phase66_layer4_trace.npz')
    p.add_argument('--metadata',default='phase25_qwen_24layers_int4.json')
    p.add_argument('--image',default='phase25_qwen_24layers_int4.bin')
    p.add_argument('--report',default='phase67_combined_mlp_report.json')
    p.add_argument('--verify-sha',action='store_true')
    a=p.parse_args()
    with np.load(a.trace,allow_pickle=False) as z:
        d={key:np.asarray(z[key],np.float32).copy() for key in z.files}
    for tag,n in [('NORM2',896),('GATE',4864),('UP',4864),('ACT',4864),('MLP_OUT',896)]:
        for prefix in ('device_','reference_'):
            key=prefix+tag
            if key not in d or d[key].shape!=(n,) or not np.all(np.isfinite(d[key])):
                raise ValueError('Invalid or missing trace: '+key)
    meta=json.loads(Path(a.metadata).read_text('utf8'))
    if Path(a.image).stat().st_size!=int(meta['image_size']):raise ValueError('Wrong INT4 image size')
    if a.verify_sha:
        with open(a.image,'rb') as f:
            if hashlib.file_digest(f,'sha256').hexdigest()!=SHA:raise ValueError('INT4 SHA mismatch')
    mats=meta['layers'][3]['matrices']
    print('PHASE67 | COMBINED LAYER4 MLP CORRECTION GATE | OFFLINE ONLY',flush=True)
    print('No firmware/COM3 changes; use captured ESP32 vectors, genuine pretrained INT4 weights.',flush=True)
    with open(a.image,'rb') as file,mmap.mmap(file.fileno(),0,access=mmap.ACCESS_READ) as mm:
        wgate,sgate=decode_matrix(mm,mats[4]);wup,sup=decode_matrix(mm,mats[5]);wdown,sdown=decode_matrix(mm,mats[6])
    g0,_=project(wgate,sgate,d['device_NORM2']);u0,_=project(wup,sup,d['device_NORM2'])
    a0=swiglu(g0,u0);o0,_=project(wdown,sdown,d['device_ACT'])
    recomputed_o0,_=project(wdown,sdown,a0)
    gate_match=metrics(g0,d['device_GATE']);up_match=metrics(u0,d['device_UP'])
    act_match=metrics(a0,d['device_ACT']);down_match=metrics(o0,d['device_MLP_OUT'])
    print('BASELINE REPLAY vs ESP32 saved trace:',flush=True)
    for name,v in [('GATE',gate_match),('UP',up_match),('ACT',act_match),('DOWN',down_match)]:
        print(f'  {name:<5} max_abs={v["max_abs"]:.9g} mixed_failures={v["failures"]}/{v["count"]}',flush=True)
    act_to_down_delta=metrics(recomputed_o0,d['device_MLP_OUT'])
    print(f'  DOWN with recomputed ACT: max_abs={act_to_down_delta["max_abs"]:.9g} failures={act_to_down_delta["failures"]}/{act_to_down_delta["count"]}',flush=True)
    # Single-source verification is intentionally strict: the reproduced Q15 stage outputs
    # must agree with the device trace. ACT may differ slightly due to expf implementations.
    if gate_match['max_abs']>1e-5 or up_match['max_abs']>1e-5 or down_match['max_abs']>1e-5:
        raise RuntimeError('BASELINE REPLAY GATE FAIL: device outputs not reproduced; do not trust candidate results')
    # Recreate reference ACT from logged GATE/UP and ensure it matches reference trace reasonably.
    ref_act_check=metrics(swiglu(d['reference_GATE'],d['reference_UP']),d['reference_ACT'])
    print(f'REFERENCE ACT RECONSTRUCTION max_abs={ref_act_check["max_abs"]:.9g}',flush=True)
    if ref_act_check['max_abs']>0.1:raise RuntimeError('Reference ACT reconstruction inconsistent')
    stages={}
    for gk,uk in itertools.product((0,1,2),repeat=2):
        g,pg=project(wgate,sgate,d['device_NORM2'],gk)
        u,pu=project(wup,sup,d['device_NORM2'],uk)
        act=swiglu(g,u)
        stages[(gk,uk)]=(g,u,act,pg,pu)
    rows=[]
    print('  gate_peels up_peels down_peels | GATEfails UPfails ACTfails DOWNfails | extra_scalar_taps',flush=True)
    for (gk,uk),(g,u,act,pg,pu) in stages.items():
        gm=metrics(g,d['reference_GATE']);um=metrics(u,d['reference_UP']);am=metrics(act,d['reference_ACT'])
        for dk in (0,1,2):
            out,pd=project(wdown,sdown,act,dk)
            dm=metrics(out,d['reference_MLP_OUT'])
            extra=gk*len(g)+uk*len(u)+dk*len(out) # one scalar tap per output row per peel
            record={'gate_peels':gk,'up_peels':uk,'down_peels':dk,
                    'gate':gm,'up':um,'act':am,'down':dm,
                    'extra_scalar_taps':extra,
                    'groups':{'gate':None if pg is None else pg['group'],
                              'up':None if pu is None else pu['group'],
                              'down':None if pd is None else pd['group']},
                    'positions':{'gate':[] if pg is None else pg['indices'].tolist(),
                                 'up':[] if pu is None else pu['indices'].tolist(),
                                 'down':[] if pd is None else pd['indices'].tolist()}}
            rows.append(record)
            print(f'       {gk:1}          {uk:1}         {dk:1}      | {gm["failures"]:4}      {um["failures"]:4}     {am["failures"]:4}     {dm["failures"]:4}       | {extra}',flush=True)
    ranked=sorted(rows,key=lambda r:(r['down']['failures'],r['extra_scalar_taps'],r['act']['failures']))
    best=ranked[0]
    passes=[r for r in rows if r['down']['failures']<=10]
    recommended=min(passes,key=lambda r:(r['extra_scalar_taps'],r['down']['failures'])) if passes else None
    report={'baseline_trace_match':{'gate':gate_match,'up':up_match,'act':act_match,'down':down_match},
            'reference_act_match':ref_act_check,'recomputed_down_vs_trace':act_to_down_delta,'candidates':rows,'best':best,'eligible':recommended,
            'warning':'Offline local MLP_OUT accuracy only; no final residual/layer result, device speed, other prompts, or 24-layer proof.'}
    Path(a.report).write_text(json.dumps(report,indent=2),encoding='utf8')
    print('\nBEST MLP_OUT:',{k:best[k] for k in ('gate_peels','up_peels','down_peels','extra_scalar_taps')},'failures',best['down']['failures'],flush=True)
    if recommended:
        print('PHASE67 OFFLINE GATE PASS: <=10 DOWN failures; candidate for ESP32 accuracy and speed verification.',flush=True)
        print('LOWEST-WORK CANDIDATE:',{k:recommended[k] for k in ('gate_peels','up_peels','down_peels','groups','positions')},flush=True)
    else:
        print('PHASE67 OFFLINE GATE FAIL: no combination reaches <=10 DOWN failures; DO NOT modify firmware.',flush=True)
    print('REPORT',a.report,flush=True)

if __name__=='__main__':run()
