"""Phase 51: scale-aware 8-layer verification, first prompt token only.
USB-C only; ESP32 computes all transformer ops; reference AFTER hardware inference.
Uses existing Phase48 firmware backwards-compatible P47 VERIFY protocol.
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
    result={}
    for pos,embed in enumerate(inputs):
        state=embed.copy()
        for idx,layer in enumerate(layers):
            m=layer['matrices']; n1=v32(mm,layer['norms'][0],D);n2=v32(mm,layer['norms'][1],D)
            bq=v32(mm,layer['biases'][0],D);bk=v32(mm,layer['biases'][1],KV);bv=v32(mm,layer['biases'][2],KV)
            x=rms(state,n1)
            q=rope(matvec(mm,m[0],x)+bq,14,pos).reshape(14,64)
            k=rope(matvec(mm,m[1],x)+bk,2,pos).reshape(2,64)
            v=(matvec(mm,m[2],x)+bv).reshape(2,64)
            keys,vals=kv[idx];keys.append(k);vals.append(v)
            ctx=np.empty((14,64),dtype=np.float32)
            for h in range(14):
                head=h//7
                scores=np.array([np.dot(q[h],keys[t][head])/8 for t in range(pos+1)],dtype=np.float64)
                probs=np.exp(scores-scores.max());probs/=probs.sum()
                ctx[h]=sum((probs[t]*vals[t][head] for t in range(pos+1)))
            residual=state+matvec(mm,m[3],ctx.reshape(-1))
            norm=rms(residual,n2)
            gate=matvec(mm,m[4],norm);up=matvec(mm,m[5],norm)
            activation=gate/(1+np.exp(-np.clip(gate,-80,80)))*up
            state=residual+matvec(mm,m[6],activation.astype(np.float32))
            result[(pos,idx)]=state.copy()
            print(f'  NumPy reference token={pos} layer={idx+1}/{len(layers)}',flush=True)
    return result

def execute(args):
    import numpy as np
    import serial
    ids,embeds=pretrained_embeddings(args.prompt,args.checkpoint,args.tokenizer_file)
    if args.first_token_only:
        print(f"CAUSAL FIRST TOKEN ONLY: ID {ids[0]}, ignoring later tokens {ids[1:]}",flush=True)
        ids,embeds=ids[:1],embeds[:1]
    if args.prepare_only:return
    meta=json.loads(Path(args.metadata).read_text(encoding='utf-8'))
    image_check(args.image,meta)
    layers=meta['layers'][:args.active_layers]
    allowed={(m['offset'],m['rows']*m['row_bytes']) for layer in layers for m in layer['matrices']}
    received={}; samples={}; final_out={}; done=False; ready=False
    with open(args.image,'rb') as f:
        mm=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ)
        try:
            with serial.Serial(args.port,115200,timeout=.35,write_timeout=8) as dev:
                print('USB-C port open; initializing (3s)...',flush=True)
                time.sleep(3);dev.reset_input_buffer()
                cmd=f'START P47 VERIFY {len(ids)} {len(layers)}\n'
                dev.write(cmd.encode('ascii'));dev.flush()
                print('SENT',repr(cmd.strip()),flush=True)
                first=time.monotonic();deadline=first+args.timeout
                while time.monotonic()<deadline:
                    raw=dev.readline()
                    if not raw:
                        if not ready and time.monotonic()-first>12:raise TimeoutError('No READY PHASE47 within 12s. Check firmware version; do not wait.')
                        continue
                    line=raw.decode('ascii',errors='replace').strip()
                    if not ready:
                        print('HANDSHAKE RX',repr(line),flush=True)
                        if line=='READY PHASE47':ready=True;continue
                        if line.startswith('ERR ') or line.startswith('READY '):raise RuntimeError('Wrong/failed firmware: '+line)
                        continue
                    if line.startswith('ERR '):raise RuntimeError('Device error: '+line)
                    if line.startswith('PHASE56_KERNEL '):
                        print(line,flush=True)
                        if 'selected=FALLBACK' in line:
                            raise RuntimeError('Selective correction rejected by ESP32 hardware speed gate; stop before model streaming.')
                        if 'selected=SELECTIVE' not in line:
                            raise RuntimeError('Unknown Phase56 kernel selection')
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
                    elif line.startswith('DONE PHASE47 VERIFY'):
                        done=True;print(line,flush=True);break
                if not done:raise TimeoutError('Incomplete Phase47 transfer')
            needed=len(ids)*len(layers)*(D//16)
            if len(received)!=needed:raise RuntimeError(f'Missing full layer vectors: {len(received)} of {needed}')
            print('Computing independent NumPy reference AFTER all ESP32 inference...',flush=True)
            expected=reference(mm,layers,embeds)
            worst_abs=0.;worst_normalized=0.;first_bad=None
            report=[]
            print('PHASE51 ALL-896-VALUE LAYER AUDIT: atol=%.6g, rtol=%.6g'%(args.atol,args.rtol),flush=True)
            for pos in range(len(ids)):
                for layer in range(len(layers)):
                    vec=np.concatenate([received[pos,layer,start] for start in range(0,D,16)])
                    ref=expected[pos,layer]
                    if not np.all(np.isfinite(ref)) or not np.all(np.isfinite(vec)):
                        raise RuntimeError('Non-finite reference/device activation')
                    error=np.abs(vec.astype(np.float64)-ref.astype(np.float64))
                    allowance=args.atol+args.rtol*np.abs(ref.astype(np.float64))
                    violation=error>allowance
                    normalized=error/np.maximum(np.abs(ref.astype(np.float64)),1.)
                    count=int(np.count_nonzero(violation))
                    absolute_max=float(error.max());mean=float(error.mean())
                    norm_max=float(normalized.max());p99=float(np.percentile(normalized,99))
                    worst_abs=max(worst_abs,absolute_max)
                    worst_normalized=max(worst_normalized,norm_max)
                    if count and first_bad is None:first_bad=(pos,layer+1,count,absolute_max)
                    print(f'  token={pos} layer={layer+1}: abs_max={absolute_max:.9g} abs_mean={mean:.9g} '
                          f'norm_max={norm_max:.9g} norm_p99={p99:.9g} '
                          f'combined_failures={count}/896',flush=True)
                    report.append(dict(token=pos,layer=layer+1,abs_max=absolute_max,
                                       abs_mean=mean,norm_max=norm_max,norm_p99=p99,
                                       combined_failures=count))
            print(f'PHASE51 WORST_ABS {worst_abs:.9g}; WORST_NORMALIZED {worst_normalized:.9g}',flush=True)
            if first_bad:
                print('PHASE51 COMBINED-TOLERANCE FAIL: first token=%d layer=%d violations=%d abs_max=%.9g'%first_bad,flush=True)
                print('Do NOT claim full-model accuracy or next-token agreement.',flush=True)
            else:
                print('PHASE51 COMBINED-TOLERANCE PASS for the evaluated layers only.',flush=True)
                print('This is NOT validation of 24 layers or output-head token selection.',flush=True)
            report_path=Path(args.report)
            report_path.write_text(json.dumps(dict(prompt=args.prompt,token_ids=ids,layers=len(layers),
                atol=args.atol,rtol=args.rtol,worst_abs=worst_abs,
                worst_normalized=worst_normalized,first_failure=first_bad,
                layer_results=report),indent=2),encoding='utf-8')
            print('REPORT SAVED',report_path,flush=True)
            if args.strict and first_bad:raise SystemExit(2)
        finally:mm.close()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prompt',default='Hi?');p.add_argument('--active-layers',type=int,default=8)
    p.add_argument('--first-token-only',action='store_true',default=True,help='One causal token for shortest test')
    p.add_argument('--atol',type=float,default=.001);p.add_argument('--rtol',type=float,default=.001)
    p.add_argument('--report',default='phase56_layer_report.json')
    p.add_argument('--port',default='COM3');p.add_argument('--timeout',type=int,default=600)
    p.add_argument('--strict',action='store_true')
    p.add_argument('--image',default='phase25_qwen_24layers_int4.bin')
    p.add_argument('--metadata',default='phase25_qwen_24layers_int4.json')
    p.add_argument('--checkpoint',default=None);p.add_argument('--tokenizer-file',default=None)
    p.add_argument('--prepare-only',action='store_true')
    args=p.parse_args()
    if not 1<=args.active_layers<=24:raise ValueError('active-layers must be 1..24')
    if args.atol<=0 or args.rtol<0:raise ValueError('Tolerances must be nonnegative, atol positive')
    execute(args)

if __name__=='__main__':main()
