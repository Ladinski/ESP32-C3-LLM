"""USB-only Phase 45 output-head proof. All model computations and greedy selection run on ESP32-C3.

HEAD_ONLY: uses pretrained token-ID 200 embedding, WITHOUT transformer; this is a
numerical test of final RMSNorm + output projection, NOT valid next-token generation.
GENERATE: original two fixed input tokens + N transformer layers + output head;
valid full-model inference requires all 24 layers and validated numerical drift.
"""
import argparse, hashlib, json, math, mmap, struct, time
from pathlib import Path

D=896
PAGE=4096

def verify_image(file,info):
    if Path(file).stat().st_size!=info['image_size']:
        raise ValueError(f'Image length mismatch for {file}')
    with open(file,'rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
    if sha!=info['sha256']:raise ValueError(f'Bad SHA256 for {file}')
    return sha

def head_reference(head_mm,state,head_meta):
    """AFTER hardware completion, calculate independent grouped-Q15 INT4 logit argmax."""
    import numpy as np
    gamma=np.frombuffer(head_mm[:D*4],dtype='<f4').copy()
    x=np.asarray(state,dtype=np.float32)
    # Match ESP32 float32 RMS accumulation for a closer independent reference.
    norm=np.float32(np.sum(x.astype(np.float64)**2)/D)
    normed=(x / np.sqrt(norm+np.float32(1e-6),dtype=np.float32)*gamma).astype(np.float32)
    steps=[];q=np.empty(D,dtype=np.int16)
    for start in range(0,D,128):
        v=normed[start:start+128]
        mx=float(np.max(np.abs(v)))
        step=np.float32(mx/32767.0 if mx else 1.0)
        q[start:start+128]=np.clip(np.rint(v/step),-32767,32767).astype(np.int16)
        steps.append(step)
    vocab=head_meta['vocab']; row_size=head_meta['row_bytes'];base=head_meta['head_offset']
    best=-1;best_logit=-float('inf');next_logit=-float('inf')
    for start in range(0,vocab,256):
        n=min(256,vocab-start)
        buf=np.frombuffer(head_mm[base+start*row_size:base+(start+n)*row_size],dtype=np.uint8).reshape(n,row_size)
        weights=buf[:,:D//2];low=(weights&15).astype(np.int16);high=(weights>>4).astype(np.int16)
        low[low>=8]-=16;high[high>=8]-=16
        unpacked=np.empty((n,D),dtype=np.int16)
        unpacked[:,0::2]=low;unpacked[:,1::2]=high
        acc=np.zeros(n,dtype=np.float32)
        for group,beg in enumerate(range(0,D,128)):
            # Dot in int64 is exact; magnitude for each group fits signed int32.
            dots=unpacked[:,beg:beg+128].astype(np.int64)@q[beg:beg+128].astype(np.int64)
            acc+=(dots.astype(np.float32)*steps[group]).astype(np.float32)
        scales=np.frombuffer(buf[:,D//2:].copy().tobytes(),dtype='<f4')
        values=acc*scales
        for i,val in enumerate(values):
            score=float(val)
            if score>best_logit:
                next_logit=best_logit;best_logit=score;best=start+i
            elif score>next_logit:next_logit=score
    return best,best_logit,next_logit

def run(args):
    import serial
    model_meta=json.loads(Path(args.metadata).read_text(encoding='utf-8'))
    head_meta=json.loads(Path(args.head_metadata).read_text(encoding='utf-8'))
    assert model_meta['model']=='Qwen/Qwen2.5-0.5B' and len(model_meta['layers'])==24
    assert head_meta['model']=='Qwen/Qwen2.5-0.5B' and head_meta['hidden']==D
    assert head_meta['head_offset']==D*4 and head_meta['row_bytes']==D//2+4
    print('MODEL VERIFIED',verify_image(args.image,model_meta),flush=True)
    print('HEAD VERIFIED',verify_image(args.head,head_meta),flush=True)
    with open(args.image,'rb') as f,open(args.head,'rb') as h:
        mm=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ)
        hm=mmap.mmap(h.fileno(),0,access=mmap.ACCESS_READ)
        try:
            line_started=False;seen_token=None;state_chunks={};last_state_chunks={};seen_done=False
            mode='HEAD_ONLY' if args.head_only else 'GENERATE '+str(args.active_layers)
            if not args.head_only:
                print(f'CAUTION: {args.active_layers} layers; genuine next-token generation requires 24, numerical drift is not resolved.',flush=True)
            with serial.Serial(args.port,115200,timeout=0.35,write_timeout=5) as dev:
                time.sleep(.7)
                dev.write(('START P45 '+mode+'\n').encode('ascii'))
                deadline=time.monotonic()+args.timeout
                while time.monotonic()<deadline:
                    raw=dev.readline()
                    if not raw:continue
                    line=raw.decode('ascii',errors='replace').strip()
                    if line=='READY PHASE45':line_started=True;print(line,flush=True);continue
                    if not line_started:continue
                    if line.startswith('ERR '):raise RuntimeError(line)
                    if line.startswith('PAGE '):
                        fields=line.split();off,n=map(int,fields[1:])
                        if len(fields)!=3 or off%PAGE or n!=min(PAGE,len(mm)-off) or off<0 or off>=len(mm):
                            raise RuntimeError('Invalid PAGE request '+line)
                        if dev.write(mm[off:off+n])!=n:raise RuntimeError('Short page write')
                    elif line.startswith('RANGE '):
                        fields=line.split();off,n=map(int,fields[1:])
                        known={(m['offset'],m['rows']*m['row_bytes']) for layer in model_meta['layers'][:args.active_layers] for m in layer['matrices']}
                        if len(fields)!=3 or (off,n) not in known:raise RuntimeError('Unexpected layer matrix request '+line)
                        for at in range(0,n,4096):
                            part=mm[off+at:off+min(n,at+4096)]
                            if dev.write(part)!=len(part):raise RuntimeError('Short RANGE write')
                    elif line.startswith('HEAD_NORM '):
                        n=int(line.split()[1]);
                        if n!=D*4:raise RuntimeError('Bad HEAD_NORM length')
                        if dev.write(hm[:n])!=n:raise RuntimeError('Short norm write')
                        print('ESP32 applying genuine pretrained final RMSNorm',flush=True)
                    elif line.startswith('HEAD_RANGE '):
                        fields=line.split();off,n=map(int,fields[1:])
                        if len(fields)!=3 or off!=head_meta['head_offset'] or n!=head_meta['vocab']*head_meta['row_bytes']:
                            raise RuntimeError('Bad HEAD_RANGE')
                        print(f'Serving {n:,} INT4 output-head bytes over USB-C. ESP32 computes all logits and argmax.',flush=True)
                        for at in range(0,n,4096):
                            part=hm[off+at:off+min(n,at+4096)]
                            if dev.write(part)!=len(part):raise RuntimeError('Short head write')
                    elif line.startswith('OUT '):
                        parts=line.split();pos,start=map(int,parts[1:3]);vals=[float(v) for v in parts[3:]]
                        if len(vals)!=16 or pos not in (0,1) or start not in range(0,D,16):
                            raise RuntimeError('Invalid hidden state')
                        state_chunks[start]=vals
                    elif line.startswith('P45_TOKEN '):
                        parts=line.split();seen_token=(int(parts[1]),float(parts[2]))
                        print(f'ESP32 SELECTED TOKEN ID {seen_token[0]} | logit {seen_token[1]:.7g}',flush=True)
                    elif line.startswith('P45_HEAD_TIME '):
                        print(f'ESP32 HEAD COMPUTE+TRANSFER TIME {int(line.split()[1])/1e6:.3f}s',flush=True)
                    elif line.startswith('TOKEN '):
                        print(line,flush=True)
                    elif line.startswith('LAYER '):
                        print('  '+line.split(' ',3)[0]+' '+line.split(' ',3)[1]+' '+line.split(' ',3)[2]+' complete',flush=True)
                    elif line.startswith('DONE PHASE45 '):
                        seen_done=True;print(line,flush=True);break
                if not seen_done:raise TimeoutError('Phase45 timed out; check firmware and serial port')
            if seen_token is None or len(state_chunks)!=D//16:
                raise RuntimeError('Device did not provide complete state/token for verification')
            hidden=[x for start in range(0,D,16) for x in state_chunks[start]]
            if not all(math.isfinite(x) for x in hidden):raise RuntimeError('Nonfinite ESP32 hidden state')
            if args.head_only:
                import numpy as np
                expected=np.frombuffer(mm[model_meta['embeddings'][1]:model_meta['embeddings'][1]+D*4],dtype='<f4')
                worst=float(np.max(np.abs(expected-np.array(hidden,dtype=np.float32))))
                if worst>1e-6:raise RuntimeError(f'ESP32 token embedding mismatch: {worst}')
                print('ESP32 INPUT EMBEDDING VERIFIED, max error',worst,flush=True)
            print('Computing independent reference AFTER ESP32 finished...',flush=True)
            predicted,logit,second=head_reference(hm,hidden,head_meta)
            print(f'OFFLINE REFERENCE token={predicted} logit={logit:.9g} margin_over_runnerup={logit-second:.6g}',flush=True)
            err=abs(logit-seen_token[1])
            if predicted!=seen_token[0] or err>0.05:
                raise RuntimeError(f'HEAD VERIFICATION FAIL: device={seen_token}, reference={(predicted,logit)}, logit_error={err:.6g}')
            print(f'PHASE45 HEAD VERIFIED: on-device final RMSNorm + {head_meta["vocab"]:,} real INT4 vocabulary rows + token selection; score abs err {err:.6g}',flush=True)
            if args.decode_token:
                try:
                    from huggingface_hub import hf_hub_download
                    from tokenizers import Tokenizer
                    path=hf_hub_download('Qwen/Qwen2.5-0.5B','tokenizer.json')
                    piece=Tokenizer.from_file(path).decode([seen_token[0]],skip_special_tokens=False)
                    print('TOKEN TEXT (for inspection only):',repr(piece),flush=True)
                except Exception as exc:
                    print('Token text decoding unavailable; numeric token ID remains verified:',exc,flush=True)
            if args.head_only: print('HEAD_ONLY is NOT generated text: input was pretrained token-200 embedding without transformer.',flush=True)
            elif args.active_layers<24:print('PARTIAL STACK: output is NOT a valid Qwen next-token prediction; remaining pretrained layers were skipped.',flush=True)
            else:print('24-LAYER TOKEN ID OBTAINED, but grouped-Q15 numerical drift across deep layers is NOT yet independently verified.',flush=True)
        finally:hm.close();mm.close()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image',default='phase25_qwen_24layers_int4.bin')
    p.add_argument('--metadata',default='phase25_qwen_24layers_int4.json')
    p.add_argument('--head',default='phase45_qwen_head_int4.bin')
    p.add_argument('--head-metadata',default='phase45_qwen_head_int4.json')
    p.add_argument('--port',default='COM3');p.add_argument('--timeout',type=int,default=2400)
    p.add_argument('--decode-token',action='store_true',help='Optional: decode selected ID after inference using pretrained tokenizer (requires tokenizers + huggingface_hub)')
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--head-only',action='store_true',help='Fast independent final projection validation only')
    group.add_argument('--generate',action='store_true',help='Run 2 fixed pretrained tokens through N layers then head')
    p.add_argument('--active-layers',type=int,default=2)
    a=p.parse_args()
    if not 1<=a.active_layers<=24:raise ValueError('active-layers must be in 1..24')
    if a.generate and a.active_layers==24:
        print('WARNING: full 24-layer grouped-Q15 inference is not precision-validated; allow as an experiment only.',flush=True)
    run(a)

if __name__=='__main__':main()
