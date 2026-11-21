"""Phase 24: serve real Qwen2.5-0.5B INT4 layer-0 weights to ESP32-C3.
Python performs NO neural work during --storage-only; standard mode independently verifies after the device finishes.
"""
import argparse, hashlib, json, math, mmap, struct, time
from pathlib import Path
import numpy as np
D,KV,FF,PAGE=896,128,4864,512
FNV0,PRIME=2166136261,16777619

def check_model(image,metadata):
    meta=json.loads(Path(metadata).read_text(encoding='utf-8'))
    if meta['model']!='Qwen/Qwen2.5-0.5B' or meta['size']!=Path(image).stat().st_size:
        raise RuntimeError('Model metadata mismatch')
    with open(image,'rb') as f: hashval=hashlib.file_digest(f,'sha256').hexdigest()
    if hashval!=meta['sha256']:raise RuntimeError('Image SHA256 mismatch')
    if meta['input_token_ids']!=[100,200] or len(meta['matrices'])!=7:raise RuntimeError('Unexpected model layout')
    print(f'MODEL PASS: official pretrained Qwen2.5-0.5B layer0 INT4 | {meta["size"]:,} bytes | SHA256 {hashval}',flush=True)
    return meta

def access_sequence(meta):
    m=meta['matrices'];o=meta['offsets'];seq=[]
    for step in range(2):
        seq.extend([(o['embedding'+str(step)],D*4),(o['norm1'],D*4),(o['bias_q_proj'],D*4),(o['bias_k_proj'],KV*4),(o['bias_v_proj'],KV*4)])
        for j in range(4):
            mat=m[j];seq.extend((mat['offset']+r*mat['row_bytes'],mat['row_bytes']) for r in range(mat['rows']))
        seq.append((o['norm2'],D*4))
        for j in range(4,7):
            mat=m[j];seq.extend((mat['offset']+r*mat['row_bytes'],mat['row_bytes']) for r in range(mat['rows']))
    return seq

def expected_storage(mm,meta):
    last=None;fetches=hits=physical=records=0;checksum=FNV0
    for addr,n in access_sequence(meta):
        rec=mm[addr:addr+n];records+=1
        for v in rec:checksum=((checksum^v)*PRIME)&0xffffffff
        at=0
        while at<n:
            pos=addr+at;aligned=pos//PAGE*PAGE
            if last==aligned:hits+=1
            else:last=aligned;fetches+=1;physical+=min(PAGE,len(mm)-aligned)
            at+=min(n-at,PAGE-pos%PAGE)
    return fetches,hits,physical,records,checksum

def f32_from(mm,offset,size):
    return np.frombuffer(mm[offset:offset+size*4],dtype='<f4').copy()

def matvec(mm,m,x):
    rows,cols=m['rows'],m['cols'];width=m['row_bytes']
    result=np.empty(rows,dtype=np.float32)
    # Process manageable groups of rows; no full-model neural operation in storage-only mode.
    for start in range(0,rows,32):
        stop=min(rows,start+32)
        segment=np.frombuffer(mm[m['offset']+start*width:m['offset']+stop*width],dtype=np.uint8).reshape(stop-start,width)
        packed=segment[:,:cols//2]
        lo=(packed&15).astype(np.int8);hi=(packed>>4).astype(np.int8)
        lo[lo>=8]-=16;hi[hi>=8]-=16
        vals=np.empty((stop-start,cols),dtype=np.float32)
        vals[:,0::2]=lo;vals[:,1::2]=hi
        scales=np.frombuffer(segment[:,cols//2:].copy().tobytes(),dtype='<f4')
        result[start:stop]=(vals@x)*scales
    return result

def rms(x,g):
    return (x/np.sqrt(np.mean(x.astype(np.float64)**2)+1e-6)*g).astype(np.float32)

def rope(x,heads,pos):
    x=x.reshape(heads,64).copy()
    inv=pos/(1_000_000.**(np.arange(32)*2/64))
    a=x[:,:32].copy();b=x[:,32:].copy()
    x[:,:32]=a*np.cos(inv)-b*np.sin(inv)
    x[:,32:]=b*np.cos(inv)+a*np.sin(inv)
    return x.reshape(-1)

def reference(mm,meta):
    m=meta['matrices'];o=meta['offsets']
    n1=f32_from(mm,o['norm1'],D);n2=f32_from(mm,o['norm2'],D)
    biases=[f32_from(mm,o['bias_'+name],size) for name,size in zip(('q_proj','k_proj','v_proj'),(D,KV,KV))]
    cache_k=[];cache_v=[];outputs=[]
    for pos in range(2):
        x=f32_from(mm,o['embedding'+str(pos)],D)
        norm=rms(x,n1)
        q=rope(matvec(mm,m[0],norm)+biases[0],14,pos).reshape(14,64)
        k=rope(matvec(mm,m[1],norm)+biases[1],2,pos).reshape(2,64)
        v=(matvec(mm,m[2],norm)+biases[2]).reshape(2,64)
        cache_k.append(k);cache_v.append(v)
        context=np.empty((14,64),dtype=np.float32)
        for h in range(14):
            kvh=h//7
            scores=np.array([float(np.dot(q[h],cache_k[t][kvh]))/8.0 for t in range(pos+1)],dtype=np.float64)
            ex=np.exp(scores-scores.max());weights=ex/ex.sum()
            context[h]=sum((weights[t]*cache_v[t][kvh] for t in range(pos+1)))
        residual=x+matvec(mm,m[3],context.reshape(-1))
        norm2=rms(residual,n2)
        gate=matvec(mm,m[4],norm2);up=matvec(mm,m[5],norm2)
        activated=(gate/(1+np.exp(-np.clip(gate,-80,80))))*up
        out=residual+matvec(mm,m[6],activated.astype(np.float32))
        outputs.append(out)
    return outputs

def run(image,meta,port,baud,timeout,storage_only):
    import serial
    from serial.serialutil import SerialException
    deadline=time.monotonic()+timeout
    with open(image,'rb') as f:
        mm=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ)
        expect=expected_storage(mm,meta)
        print(f'Predicted page fetches={expect[0]}, physical bytes={expect[2]:,}; INT4 row reads={expect[3]-12}',flush=True)
        chunks={};stats=None;done=False;ready=False;dev=None;start=0;reqs=0;physical=0
        try:
            while time.monotonic()<deadline and not done:
                try:
                    if dev is None:
                        dev=serial.Serial(port,baudrate=baud,timeout=.3,write_timeout=3)
                        time.sleep(.7)
                    if not ready and time.monotonic()-start>1.5:
                        dev.write(b'START\n');start=time.monotonic()
                    raw=dev.readline()
                    if not raw:continue
                    line=raw.decode('ascii',errors='replace').strip()
                    if not line:continue
                    if line=='READY PHASE24':ready=True;print(line,flush=True);continue
                    if not ready:continue
                    if line.startswith('PAGE '):
                        parts=line.split()
                        if len(parts)!=3:raise RuntimeError('Malformed PAGE '+line)
                        off,n=map(int,parts[1:])
                        if off%PAGE or n!=min(PAGE,len(mm)-off) or n<=0 or off>=len(mm):
                            raise RuntimeError('Invalid ESP32 storage request '+line)
                        dev.write(mm[off:off+n]);reqs+=1;physical+=n
                        if reqs%2048==0:print(f'  served {reqs:,} ESP32 page requests',flush=True)
                    elif line.startswith('OUT '):
                        parts=line.split();pos=int(parts[1]);start_idx=int(parts[2]);vals=[float(v) for v in parts[3:]]
                        if pos not in (0,1) or start_idx not in range(0,D,16) or len(vals)!=16 or not all(map(math.isfinite,vals)):
                            raise RuntimeError('Malformed ESP32 output '+line)
                        key=(pos,start_idx)
                        if key in chunks:raise RuntimeError('Duplicate output chunk')
                        chunks[key]=vals
                    elif line.startswith('STATS '):
                        pieces=line.split()
                        if len(pieces)!=8:raise RuntimeError('Malformed stats '+line)
                        stats=tuple(map(int,pieces[1:5]))+(int(pieces[5],16),)+tuple(map(int,pieces[6:]))
                        print(line,flush=True)
                    elif line.startswith('TOKEN '):print(line,flush=True)
                    elif line.startswith('DONE PHASE24'):print(line,flush=True);done=True
                    elif line.startswith('CONFIG '):print(line,flush=True)
                    elif line.startswith('ERR '):raise RuntimeError('Device error: '+line)
                except (SerialException,OSError) as e:
                    if ready:raise RuntimeError('USB disconnected mid-run') from e
                    if dev:
                        try:dev.close()
                        except Exception:pass
                    dev=None;time.sleep(.3)
            if not done:raise TimeoutError('No Phase 24 DONE; ensure flash completed and monitor closed')
            if stats is None or stats[:5]!=expect:raise RuntimeError(f'Storage stats mismatch: device {stats} vs expected {expect}')
            if (reqs,physical)!=(expect[0],expect[2]):raise RuntimeError('Host physical transfer count mismatch')
            if stats[5]<32768 or stats[6]<256:raise RuntimeError('ESP32 heap or stack below safety threshold')
            for pos in range(2):
                if any((pos,i) not in chunks for i in range(0,D,16)):raise RuntimeError(f'Missing output chunks at pos={pos}')
            print(f'STORAGE PASS: {reqs:,} ESP32-driven page fetches, {physical:,} bytes',flush=True)
            if not storage_only:
                print('Computing NumPy reference AFTER ESP32 inference...',flush=True)
                answers=reference(mm,meta)
                for pos in range(2):
                    devout=np.array(sum((chunks[pos,i] for i in range(0,D,16)),[]),dtype=np.float32)
                    err=np.abs(devout-answers[pos]);maximum=float(err.max());scale=np.maximum(1.0,np.abs(answers[pos]))
                    rel=float((err/scale).max())
                    print(f'Token {pos}: max absolute error {maximum:.6g}; max scaled error {rel:.6g}',flush=True)
                    if not np.allclose(devout,answers[pos],rtol=.006,atol=.03):
                        ix=np.argmax(err);raise RuntimeError(f'Token {pos}: mismatch at {ix}: ESP={devout[ix]} PC={answers[pos][ix]}')
                print('PHASE 24 PASS: genuine pretrained Qwen2.5-0.5B layer 0, two-token causal attention, ESP32 INT4 paging, NumPy reference',flush=True)
            else:print('PHASE 24 STORAGE-ONLY PASS: no PC neural inference during run',flush=True)
        finally:
            if dev:dev.close()
            mm.close()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image',default='phase24_qwen_layer0_int4.bin');p.add_argument('--metadata',default='phase24_qwen_layer0_int4.json')
    p.add_argument('--port',default='COM3');p.add_argument('--baud',type=int,default=115200)
    p.add_argument('--timeout',type=float,default=1800)
    p.add_argument('--storage-only',action='store_true');p.add_argument('--check-only',action='store_true')
    args=p.parse_args()
    meta=check_model(args.image,args.metadata)
    if args.check_only:return
    run(args.image,meta,args.port,args.baud,args.timeout,args.storage_only)
if __name__=='__main__':main()
