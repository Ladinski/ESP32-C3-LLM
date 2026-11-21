"""Phase 47: verify real prompt hidden states against INT4 NumPy reference AFTER ESP32 runs.
USB-C only; ESP32 computes every transformer operation; no head stream in this mode.
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
            worst=0.;first_bad=None
            print('FULL ACTIVATION COMPARISON (all 896 features at every layer):')
            for pos in range(len(ids)):
                for layer in range(len(layers)):
                    vec=np.concatenate([received[pos,layer,start] for start in range(0,D,16)])
                    err=np.abs(vec-expected[pos,layer]);mx=float(err.max());worst=max(worst,mx)
                    n1=int(np.sum(err>0.001));n2=int(np.sum(err>0.01))
                    if mx>args.max_error and first_bad is None:first_bad=(pos,layer+1,mx)
                    print(f'  token={pos} layer={layer+1}: max={mx:.9g}, mean={float(err.mean()):.9g}, >0.001={n1}, >0.01={n2}',flush=True)
            print(f'PHASE47 WORST ERROR {worst:.9g}; limit {args.max_error}',flush=True)
            if first_bad:
                print('PHASE47 ACCURACY DRIFT DETECTED: first failure token=%d layer=%d max_error=%.9g'%first_bad,flush=True)
                print('Do NOT treat full 24-layer grouped-Q15 text as numerically verified.',flush=True)
            else:
                print('PHASE47 PRECISION PASS for evaluated layers at absolute error threshold.',flush=True)
            if args.strict and first_bad:raise SystemExit(2)
        finally:mm.close()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prompt',default='Hi?');p.add_argument('--active-layers',type=int,default=4)
    p.add_argument('--port',default='COM3');p.add_argument('--timeout',type=int,default=600)
    p.add_argument('--max-error',type=float,default=.001);p.add_argument('--strict',action='store_true')
    p.add_argument('--image',default='phase25_qwen_24layers_int4.bin')
    p.add_argument('--metadata',default='phase25_qwen_24layers_int4.json')
    p.add_argument('--checkpoint',default=None);p.add_argument('--tokenizer-file',default=None)
    p.add_argument('--prepare-only',action='store_true')
    args=p.parse_args()
    if not 1<=args.active_layers<=24:raise ValueError('active-layers must be 1..24')
    execute(args)

if __name__=='__main__':main()
