"""Phase 49: post-hardware detailed SwiGLU numerical root-cause diagnostic.
Only the ESP32 performs inference; NumPy compares AFTER hardware execution.
"""
import argparse, hashlib, json, math, mmap, time
from pathlib import Path

D=896; KV=128; PAGE=4096

def image_check(path, meta):
    if Path(path).stat().st_size != meta['image_size']: raise RuntimeError('Wrong model image size')
    with open(path,'rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
    if digest!=meta['sha256']: raise RuntimeError('Model SHA256 mismatch')
    print('MODEL VERIFIED',digest,flush=True)

def pretrained_embeddings(prompt, checkpoint, tokenizer_file):
    import numpy as np
    from tokenizers import Tokenizer
    from huggingface_hub import hf_hub_download
    from export_phase45_head import BFloatReader, REPO
    tok=Tokenizer.from_file(tokenizer_file or hf_hub_download(REPO,'tokenizer.json'))
    ids=tok.encode(prompt,add_special_tokens=False).ids
    if not 1<=len(ids)<=2: raise RuntimeError(f'Only 1-2 prompt tokens supported; got {ids}')
    with BFloatReader(checkpoint or hf_hub_download(REPO,'model.safetensors')) as source:
        rows=[np.asarray(source.get_row('model.embed_tokens.weight',i),dtype='<f4').copy() for i in ids]
    print('REAL PROMPT',repr(prompt),'TOKEN IDS',ids,flush=True)
    return ids,rows

def v32(mm,off,n):
    import numpy as np
    return np.frombuffer(mm[off:off+4*n],dtype='<f4').copy()

def matvec(mm,m,x):
    import numpy as np
    rows,cols=m['rows'],m['cols'];width=m['row_bytes']
    output=np.empty(rows,dtype=np.float32)
    for start in range(0,rows,32):
        count=min(32,rows-start)
        b=np.frombuffer(mm[m['offset']+start*width:m['offset']+(start+count)*width],dtype=np.uint8).reshape(count,width)
        pack=b[:,:cols//2]
        lo=(pack&15).astype(np.int8);hi=(pack>>4).astype(np.int8)
        lo[lo>=8]-=16;hi[hi>=8]-=16
        w=np.empty((count,cols),dtype=np.float32)
        w[:,0::2]=lo;w[:,1::2]=hi
        scales=np.frombuffer(b[:,cols//2:].copy().tobytes(),dtype='<f4')
        output[start:start+count]=(w@x)*scales
    return output

def rms(x,g):
    import numpy as np
    return (x/np.sqrt(np.mean(x.astype(np.float64)**2)+1e-6)*g).astype(np.float32)

def rope(x,heads,pos):
    import numpy as np
    x=x.reshape(heads,64).copy()
    ang=pos/(1_000_000.**(np.arange(32)*2/64))
    a=x[:,:32].copy();b=x[:,32:].copy()
    x[:,:32]=a*np.cos(ang)-b*np.sin(ang)
    x[:,32:]=b*np.cos(ang)+a*np.sin(ang)
    return x.reshape(-1)

def reference(mm,layers,inputs):
    import numpy as np
    kv=[([],[]) for _ in layers]
    result={}; traces={}
    for pos,embed in enumerate(inputs):
        state=embed.copy()
        for idx,layer in enumerate(layers):
            m=layer['matrices']; n1=v32(mm,layer['norms'][0],D);n2=v32(mm,layer['norms'][1],D)
            bq=v32(mm,layer['biases'][0],D);bk=v32(mm,layer['biases'][1],KV);bv=v32(mm,layer['biases'][2],KV)
            x=rms(state,n1)
            if pos==0 and idx==2:traces["NORM1"]=x.copy()
            q=rope(matvec(mm,m[0],x)+bq,14,pos).reshape(14,64)
            k=rope(matvec(mm,m[1],x)+bk,2,pos).reshape(2,64)
            v=(matvec(mm,m[2],x)+bv).reshape(2,64)
            if pos==0 and idx==2:traces["VALUE"]=v.reshape(-1).copy()
            keys,vals=kv[idx];keys.append(k);vals.append(v)
            ctx=np.empty((14,64),dtype=np.float32)
            for h in range(14):
                head=h//7
                scores=np.array([np.dot(q[h],keys[t][head])/8 for t in range(pos+1)],dtype=np.float64)
                probs=np.exp(scores-scores.max());probs/=probs.sum()
                ctx[h]=sum((probs[t]*vals[t][head] for t in range(pos+1)))
            attn_out=matvec(mm,m[3],ctx.reshape(-1))
            residual=state+attn_out
            if pos==0 and idx==2:
                traces["CONTEXT"]=ctx.reshape(-1).copy()
                traces["ATTN_OUT"]=attn_out.copy()
                traces["RESID1"]=residual.copy()
            norm=rms(residual,n2)
            if pos==0 and idx==2:traces["NORM2"]=norm.copy()
            gate=matvec(mm,m[4],norm);up=matvec(mm,m[5],norm)
            activation=gate/(1+np.exp(-np.clip(gate,-80,80)))*up
            mlp_out=matvec(mm,m[6],activation.astype(np.float32))
            state=residual+mlp_out
            if pos==0 and idx==2:
                traces["GATE"]=gate.copy()
                traces["UP"]=up.copy()
                traces["ACT"]=activation.astype(np.float32).copy()
                traces["MLP_OUT"]=mlp_out.copy()
            result[(pos,idx)]=state.copy()
            print(f'  NumPy reference token={pos} layer={idx+1}/{len(layers)}',flush=True)
    return result,traces

def execute(args):
    import numpy as np
    import serial
    ids,embeds=pretrained_embeddings(args.prompt,args.checkpoint,args.tokenizer_file)
    # Causal attention: token 0 cannot depend on future tokens.
    ids=ids[:1];embeds=embeds[:1]
    print('PHASE48: tracing token 0 only, skipping later prompt tokens',flush=True)
    if args.prepare_only:return
    meta=json.loads(Path(args.metadata).read_text(encoding='utf-8'))
    image_check(args.image,meta)
    layers=meta['layers'][:args.active_layers]
    allowed={(m['offset'],m['rows']*m['row_bytes']) for layer in layers for m in layer['matrices']}
    received={}; trace_packets={}; samples={}; final_out={}; done=False; ready=False
    with open(args.image,'rb') as f:
        mm=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ)
        try:
            with serial.Serial(args.port,115200,timeout=.35,write_timeout=8) as dev:
                print('USB-C port open; initializing (3s)...',flush=True)
                time.sleep(3);dev.reset_input_buffer()
                cmd=f'START P48 TRACE {len(ids)} {len(layers)}\n'
                dev.write(cmd.encode('ascii'));dev.flush()
                print('SENT',repr(cmd.strip()),flush=True)
                first=time.monotonic();deadline=first+args.timeout
                while time.monotonic()<deadline:
                    raw=dev.readline()
                    if not raw:
                        if not ready and time.monotonic()-first>12:raise TimeoutError('No READY PHASE48 within 12s. Reflash Phase48 firmware; do not wait.')
                        continue
                    line=raw.decode('ascii',errors='replace').strip()
                    if not ready:
                        print('HANDSHAKE RX',repr(line),flush=True)
                        if line=='READY PHASE48':ready=True;continue
                        if line.startswith('ERR ') or line.startswith('READY '):raise RuntimeError('Wrong/failed firmware: '+line)
                        continue
                    if line.startswith('ERR '):raise RuntimeError('Device error: '+line)
                    parts=line.split()
                    if line.startswith('P46_EMBED '):
                        if len(parts)!=3:raise RuntimeError('Invalid embedding request')
                        pos,n=map(int,parts[1:]);
                        if not 0<=pos<len(ids) or n!=D*4:raise RuntimeError('Unexpected embedding request')
                        payload=embeds[pos].tobytes()
                        if dev.write(payload)!=len(payload):raise RuntimeError('Short embedding write')
                        print('  ESP32 embedding supplied for token',pos,flush=True)
                    elif line.startswith('PAGE '):
                        if len(parts)!=3:raise RuntimeError('Bad PAGE')
                        off,n=map(int,parts[1:])
                        if off<0 or off>=len(mm) or off%PAGE or n!=min(PAGE,len(mm)-off):raise RuntimeError('Unexpected PAGE request')
                        if dev.write(mm[off:off+n])!=n:raise RuntimeError('Short PAGE')
                    elif line.startswith('RANGE '):
                        if len(parts)!=3:raise RuntimeError('Bad RANGE')
                        off,n=map(int,parts[1:])
                        if (off,n) not in allowed:raise RuntimeError('Unexpected RANGE request')
                        for at in range(0,n,4096):
                            chunk=mm[off+at:off+min(n,at+4096)]
                            if dev.write(chunk)!=len(chunk):raise RuntimeError('Short RANGE write')
                    elif line.startswith('P48_STAGE '):
                        if len(parts)!=21:raise RuntimeError('Malformed trace packet: '+line[:80])
                        stage=parts[1];pos,layer,start=map(int,parts[2:5])
                        sizes={'NORM1':896,'VALUE':128,'CONTEXT':896,'ATTN_OUT':896,'RESID1':896,'NORM2':896,'GATE':4864,'UP':4864,'ACT':4864,'MLP_OUT':896}
                        if stage not in sizes or pos!=0 or layer!=2 or start not in range(0,sizes[stage],16):
                            raise RuntimeError('Unexpected trace packet location')
                        vals=np.array([float(v) for v in parts[5:]],dtype=np.float32)
                        key=(stage,start)
                        if key in trace_packets or len(vals)!=16 or not np.all(np.isfinite(vals)):
                            raise RuntimeError('Invalid/duplicate trace packet')
                        trace_packets[key]=vals
                    elif line.startswith('LAYER_FULL '):
                        if len(parts)!=20:raise RuntimeError('Incomplete full layer packet')
                        pos,layer,start=map(int,parts[1:4]);vals=np.array([float(v) for v in parts[4:]],dtype=np.float32)
                        key=(pos,layer,start)
                        if pos not in range(len(ids)) or layer not in range(len(layers)) or start not in range(0,D,16) or key in received or not np.all(np.isfinite(vals)):
                            raise RuntimeError('Unexpected layer packet')
                        received[key]=vals
                    elif line.startswith('LAYER '):
                        pos,layer=map(int,parts[1:3]);samples[pos,layer]=True
                        print(f'  ESP32 token={pos}, layer={layer+1}/{len(layers)} complete',flush=True)
                    elif line.startswith('OUT '):
                        pos,start=map(int,parts[1:3]);final_out[pos,start]=parts[3:]
                    elif line.startswith('TOKEN '): print(line,flush=True)
                    elif line.startswith('DONE PHASE48 TRACE'):
                        done=True;print(line,flush=True);break
                if not done:raise TimeoutError('Incomplete Phase47 transfer')
            needed=len(ids)*len(layers)*(D//16)
            if len(received)!=needed:raise RuntimeError(f'Missing full layer vectors: {len(received)} of {needed}')
            print('Computing independent NumPy reference AFTER all ESP32 inference...',flush=True)
            expected,reference_traces=reference(mm,layers,embeds)
            worst=0.;first_bad=None
            print('FULL ACTIVATION COMPARISON (all 896 features at every layer):')
            for pos in range(len(ids)):
                for layer in range(len(layers)):
                    vec=np.concatenate([received[pos,layer,start] for start in range(0,D,16)])
                    err=np.abs(vec-expected[pos,layer]);mx=float(err.max());worst=max(worst,mx)
                    n1=int(np.sum(err>0.001));n2=int(np.sum(err>0.01))
                    if mx>args.max_error and first_bad is None:first_bad=(pos,layer+1,mx)
                    print(f'  token={pos} layer={layer+1}: max={mx:.9g}, mean={float(err.mean()):.9g}, >0.001={n1}, >0.01={n2}',flush=True)
            print('PHASE48 STAGE-BY-STAGE TRACE (token=0 layer=3):',flush=True)
            first_trace_failure=None
            for stage in ['NORM1','VALUE','CONTEXT','ATTN_OUT','RESID1','NORM2','GATE','UP','ACT','MLP_OUT']:
                exp=reference_traces[stage]
                expected_chunks=len(exp)//16
                keys=[(stage,k) for k in range(0,len(exp),16)]
                if not all(k in trace_packets for k in keys):
                    missing=[k for k in keys if k not in trace_packets]
                    raise RuntimeError(f'Missing {len(missing)}/{expected_chunks} trace chunks for {stage}')
                device=np.concatenate([trace_packets[k] for k in keys])
                delta=np.abs(device-exp)
                peak=float(np.max(delta));average=float(np.mean(delta))
                count=int(np.count_nonzero(delta>args.max_error))
                print(f'  {stage:9s} len={len(exp):4d} max={peak:.9g} mean={average:.9g} >{args.max_error:g}={count}',flush=True)
                if first_trace_failure is None and peak>args.max_error:first_trace_failure=(stage,peak,count)
            # Phase 49: after-device-only root-cause triage. This code runs on PC
            # strictly AFTER the ESP32 has completed all neural operations.
            dev_stage={name:np.concatenate([trace_packets[(name,k)] for k in range(0,len(reference_traces[name]),16)])
                       for name in ('GATE','UP','ACT')}
            dg,du,da=(dev_stage[k] for k in ('GATE','UP','ACT'))
            rg,ru,ra=(reference_traces[k] for k in ('GATE','UP','ACT'))
            with np.errstate(over='ignore',under='ignore',invalid='ignore'):
                # Compare the device's ACT against the same mathematical formula
                # applied to the values measured *on device* immediately before ACT.
                # This is diagnostic replay, not inference or token selection.
                predicted=(dg/(np.float32(1)+np.exp(-dg))*du).astype(np.float32)
                predicted64=(dg.astype(np.float64)/(1+np.exp(-np.clip(dg.astype(np.float64),-700,700)))
                             *du.astype(np.float64)).astype(np.float32)
            act_err=np.abs(da-ra)
            intrinsic_err=np.abs(da-predicted)
            upstream_err=np.abs(predicted-ra)
            calc64_err=np.abs(da-predicted64)
            top=np.argsort(act_err)[::-1][:12]
            print('PHASE49 ACT ROOT-CAUSE TRIAGE (post-inference only):',flush=True)
            print('  Device ACT vs recomputed FP32 ACT from DEVICE GATE/UP: '
                  f'max={float(intrinsic_err.max()):.9g}, count>0.001={int((intrinsic_err>args.max_error).sum())}',flush=True)
            print('  Recomputed ACT from DEVICE inputs vs NumPy reference ACT: '
                  f'max={float(upstream_err.max()):.9g}, count>0.001={int((upstream_err>args.max_error).sum())}',flush=True)
            print('  Device ACT vs FP64 replay from DEVICE inputs: '
                  f'max={float(calc64_err.max()):.9g}',flush=True)
            print('  Top ACT mismatches: index | device_gate | ref_gate | device_up | ref_up | device_act | ref_act | device-vs-replay | replay-vs-reference')
            for j in top:
                j=int(j)
                print(f'  {j:4d} | {dg[j]:.8g} | {rg[j]:.8g} | {du[j]:.8g} | {ru[j]:.8g} | '
                      f'{da[j]:.8g} | {ra[j]:.8g} | {intrinsic_err[j]:.8g} | {upstream_err[j]:.8g}',flush=True)
            if float(intrinsic_err.max())<=args.max_error and float(upstream_err.max())>args.max_error:
                print('PHASE49 FINDING: ACT is consistent with ESP32 gate/up inputs; amplified upstream differences dominate. No evidence for a SwiGLU implementation defect.',flush=True)
            elif float(intrinsic_err.max())>args.max_error:
                print('PHASE49 FINDING: ACT differs from FP32 replay of the ESP32 gate/up inputs; investigate float rounding/overflow or implementation.',flush=True)
            else:
                print('PHASE49 FINDING: ACT replay does not explain the full observed discrepancy; inspect numerical formatting/reference differences.',flush=True)
            if args.trace_save:
                saved={'device_'+k:dev_stage[k] for k in ('GATE','UP','ACT')}
                saved.update({'reference_'+k:reference_traces[k] for k in ('GATE','UP','ACT')})
                np.savez_compressed(args.trace_save,**saved)
                print('PHASE49 SAVED TRACE:',args.trace_save,flush=True)
            if first_trace_failure:
                stage,peak,count=first_trace_failure
                print(f'PHASE48 FIRST STAGE EXCEEDING THRESHOLD: {stage}, max={peak:.9g}, affected={count}',flush=True)
                print('Interpretation: this is the first *measured* divergence; the cause may lie upstream (e.g. activation quantization).',flush=True)
            else:print('PHASE48 NO STAGE EXCEEDS THRESHOLD in token0 layer3',flush=True)
            print(f'PHASE48 WORST LAYER ERROR {worst:.9g}; limit {args.max_error}',flush=True)
            if first_bad:
                print('PHASE48 ACCURACY DRIFT DETECTED: first failure token=%d layer=%d max_error=%.9g'%first_bad,flush=True)
                print('Do NOT treat full 24-layer grouped-Q15 text as numerically verified.',flush=True)
            else:
                print('PHASE48 PRECISION PASS for evaluated layers at absolute error threshold.',flush=True)
            if args.strict and first_bad:raise SystemExit(2)
        finally:mm.close()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prompt',default='Hi?');p.add_argument('--active-layers',type=int,default=3)
    p.add_argument('--port',default='COM3');p.add_argument('--timeout',type=int,default=600)
    p.add_argument('--max-error',type=float,default=.001);p.add_argument('--strict',action='store_true')
    p.add_argument('--trace-save',default='phase49_act_trace.npz',help='Saved PC-side diagnostics; no model weights or personal data')
    p.add_argument('--image',default='phase25_qwen_24layers_int4.bin')
    p.add_argument('--metadata',default='phase25_qwen_24layers_int4.json')
    p.add_argument('--checkpoint',default=None);p.add_argument('--tokenizer-file',default=None)
    p.add_argument('--prepare-only',action='store_true')
    args=p.parse_args()
    if args.active_layers!=3:raise ValueError('Phase48 traces exactly 3 layers')
    execute(args)

if __name__=='__main__':main()
