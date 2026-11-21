"""Phase 70: two-token causal prompt, full 24-layer ESP32-C3 next-token and offline audit.
Uses the ALREADY FLASHED Phase 68 firmware; no firmware modifications.
The host streams immutable pretrained weights and embedding rows only during inference.
Independent NumPy model inference begins AFTER ESP32 has finished.
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

def prepare_prompt(args):
    """Tokenize text and copy pretrained embedding rows, not neural inference."""
    from tokenizers import Tokenizer
    from huggingface_hub import hf_hub_download
    from export_phase45_head import BFloatReader, REPO
    import numpy as np
    tokenizer_path=args.tokenizer_file or hf_hub_download(REPO,'tokenizer.json')
    checkpoint_path=args.checkpoint or hf_hub_download(REPO,'model.safetensors')
    tokenizer=Tokenizer.from_file(str(tokenizer_path))
    ids=tokenizer.encode(args.prompt,add_special_tokens=False).ids
    if len(ids)!=2:
        raise ValueError(f'Phase 70 requires EXACTLY 2 Qwen tokens, got {len(ids)}: {ids}. Try --prompt "Hi?".')
    with BFloatReader(checkpoint_path) as source:
        if 'model.embed_tokens.weight' not in source.meta:raise ValueError('Checkpoint embedding tensor missing')
        if source.meta['model.embed_tokens.weight']['shape']!=[151936,D]:raise ValueError('Unexpected checkpoint embedding shape')
        rows=[np.asarray(source.get_row('model.embed_tokens.weight',token),dtype='<f4').tobytes() for token in ids]
    if any(len(row)!=D*4 for row in rows):raise RuntimeError('Embedding row length invalid')
    print('PROMPT',repr(args.prompt),'TOKENS',ids,flush=True)
    print('EMBEDDINGS: official checkpoint lookup/format conversion only; no host neural calculations.',flush=True)
    return tokenizer,ids,rows

def run(args):
    tokenizer,ids,embeddings=prepare_prompt(args)
    if args.prepare_only:return
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
            line_started=False;seen_token=None;state_chunks={0:{},1:{}};seen_done=False;applied=None;head_seen=False
            requested_positions=[];completed_positions=[];layer_progress={0:set(),1:set()}
            mode='GENERATE '+str(len(ids))+' '+str(args.active_layers)
            print(f'CAUTION: {args.active_layers} layers. Full-model next-token prediction requires 24 layers; deep grouped-Q15 drift is unverified.',flush=True)
            with serial.Serial(args.port,115200,timeout=0.35,write_timeout=5) as dev:
                time.sleep(.7)
                dev.write(('START P68 '+mode+'\n').encode('ascii'))
                deadline=time.monotonic()+args.timeout
                while time.monotonic()<deadline:
                    raw=dev.readline()
                    if not raw:continue
                    line=raw.decode('ascii',errors='replace').strip()
                    if line=='READY PHASE68':line_started=True;print(line,flush=True);continue
                    if not line_started:continue
                    if line.startswith('ERR '):raise RuntimeError(line)
                    if line.startswith('P46_EMBED '):
                        fields=line.split()
                        if len(fields)!=3:raise RuntimeError('Bad embedding request '+line)
                        pos,n=map(int,fields[1:])
                        if pos not in range(len(ids)) or n!=D*4 or pos!=len(requested_positions):raise RuntimeError('Unexpected or repeated embedding request '+line)
                        if pos>0 and (completed_positions!=list(range(pos)) or len(state_chunks[pos-1])!=D//16):
                            raise RuntimeError('Previous token hidden-state stream incomplete before next embedding: '+line)
                        requested_positions.append(pos)
                        if dev.write(embeddings[pos])!=n:raise RuntimeError('Short embedding write')
                        print(f'ESP32 received pretrained embedding for position {pos}, token {ids[pos]}',flush=True)
                    elif line.startswith('PAGE '):
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
                    elif line.startswith('PHASE68_APPLIED '):
                        vals={k:int(v) for k,v in (field.split('=',1) for field in line.split()[1:])}
                        expected=len(ids)*D
                        if vals.get('rows')!=expected or not (0<=vals.get('group',-1)<38) or not (0<=vals.get('outlier',-1)<128):
                            raise RuntimeError(f'Correction proof failed: {line}; expected rows={expected}')
                        applied=vals
                        print('DEVICE VERIFIED layer-3 corrected rows',expected,vals,flush=True)
                    elif line.startswith('HEAD_NORM '):
                        if applied is None:raise RuntimeError('Head requested before valid device correction proof')
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
                        if pos not in completed_positions:raise RuntimeError('OUT chunk before TOKEN COMPLETE: '+line)
                        if start in state_chunks[pos]:raise RuntimeError('Duplicate OUT chunk '+line)
                        state_chunks[pos][start]=vals
                    elif line.startswith('P45_TOKEN '):
                        parts=line.split();seen_token=(int(parts[1]),float(parts[2]))
                        print(f'ESP32 SELECTED TOKEN ID {seen_token[0]} | logit {seen_token[1]:.7g}',flush=True)
                    elif line.startswith('P45_HEAD_TIME '):
                        print(f'ESP32 HEAD COMPUTE+TRANSFER TIME {int(line.split()[1])/1e6:.3f}s',flush=True)
                    elif line.startswith('TOKEN '):
                        fields=line.split()
                        if len(fields)==3 and fields[2]=='COMPLETE':
                            pos=int(fields[1])
                            # Firmware emits TOKEN <pos> COMPLETE BEFORE streaming OUT chunks.
                            # Verify layer completion here; hidden stream is checked at the
                            # next embedding request and after DONE PHASE68 GENERATE.
                            if pos!=len(completed_positions) or len(layer_progress[pos])!=24:
                                raise RuntimeError('Incomplete causal token or missing 24 layers: '+line)
                            completed_positions.append(pos)
                        print(line,flush=True)
                    elif line.startswith('LAYER '):
                        fields=line.split()
                        pos,layer=int(fields[1]),int(fields[2])
                        if pos not in layer_progress or layer not in range(24) or layer!=len(layer_progress[pos]):
                            raise RuntimeError('Out-of-order transformer layer '+line)
                        layer_progress[pos].add(layer)
                        print(f'  ESP32 token={pos} layer={layer+1}/24 complete',flush=True)
                    elif line.startswith('DONE PHASE68 '):
                        seen_done=True;print(line,flush=True);break
                if not seen_done:raise TimeoutError('Phase68 timed out; check firmware and serial port')
            if applied is None:raise RuntimeError('Missing PHASE68_APPLIED device proof')
            if requested_positions!=[0,1] or completed_positions!=[0,1]:
                raise RuntimeError(f'Incomplete two-token causal execution: embeddings={requested_positions}, done={completed_positions}')
            if seen_token is None or any(len(state_chunks[pos])!=D//16 for pos in (0,1)):
                raise RuntimeError('Device did not provide complete state/token for verification')
            all_hidden=[[x for start in range(0,D,16) for x in state_chunks[pos][start]] for pos in (0,1)]
            hidden=all_hidden[1]
            if not all(math.isfinite(x) for x in hidden):raise RuntimeError('Nonfinite ESP32 hidden state')
            print('PHASE70 DEVICE CAUSAL PATH PASS: token positions 0,1; 24 layers per position; 2 complete hidden states',flush=True)
            print('Computing independent output-head check AFTER ESP32 finished...',flush=True)
            predicted,logit,second=head_reference(hm,hidden,head_meta)
            print(f'OFFLINE REFERENCE token={predicted} logit={logit:.9g} margin_over_runnerup={logit-second:.6g}',flush=True)
            err=abs(logit-seen_token[1])
            if predicted!=seen_token[0] or err>0.05:
                raise RuntimeError(f'HEAD VERIFICATION FAIL: device={seen_token}, reference={(predicted,logit)}, logit_error={err:.6g}')
            print(f'PHASE45 HEAD VERIFIED: on-device final RMSNorm + {head_meta["vocab"]:,} real INT4 vocabulary rows + token selection; score abs err {err:.6g}',flush=True)
            if True:
                try:
                    piece=tokenizer.decode([seen_token[0]],skip_special_tokens=False)
                    print('TOKEN TEXT (for inspection only):',repr(piece),flush=True)
                except Exception as exc:
                    print('Token text decoding unavailable; numeric token ID remains verified:',exc,flush=True)
            # Preserve device results BEFORE attempting the expensive reference, so a crash
            # or interruption never forces another ESP32 run. No host inference occurred
            # until the ESP32 announced DONE PHASE68 GENERATE.
            import numpy as np
            from phase69_reference_math import reference
            final_device=np.asarray(hidden,dtype=np.float32)
            embed_vectors=[np.frombuffer(row,dtype='<f4').copy() for row in embeddings]
            np.savez_compressed(args.capture,device_final=final_device,device_positions=np.asarray(all_hidden,dtype=np.float32),
                embeddings=np.stack(embed_vectors),token_ids=np.asarray(ids,dtype=np.int64),
                device_token=np.int64(seen_token[0]),device_token_logit=np.float32(seen_token[1]))
            print('PHASE70 DEVICE CAPTURE SAVED',args.capture,flush=True)
            print('PHASE70 starting independent two-position, 24-layer FP32 INT4 reference AFTER device DONE...',flush=True)
            expected=reference(mm,model_meta['layers'][:24],embed_vectors)
            ref_state=np.asarray(expected[len(ids)-1,23],dtype=np.float32)
            np.savez_compressed(args.reference_capture,reference_final=ref_state,reference_positions=np.stack([expected[pos,23] for pos in (0,1)]),
                device_final=final_device,token_ids=np.asarray(ids,dtype=np.int64))
            print('PHASE70 REFERENCE CAPTURE SAVED',args.reference_capture,flush=True)
            for pos in (0,1):
                device_vec=np.asarray(all_hidden[pos],dtype=np.float64)
                ref_vec=np.asarray(expected[pos,23],dtype=np.float64)
                dist=np.abs(device_vec-ref_vec)
                errs=int(np.count_nonzero(dist>args.atol+args.rtol*np.abs(ref_vec)))
                print(f'PHASE70 POSITION {pos} FINAL HIDDEN: mixed_failures={errs}/896 max_abs={dist.max():.9g} mean_abs={dist.mean():.9g}',flush=True)
            difference=np.abs(final_device.astype(np.float64)-ref_state.astype(np.float64))
            tolerance=args.atol+args.rtol*np.abs(ref_state.astype(np.float64))
            violations=int(np.count_nonzero(difference>tolerance))
            norm=float(np.max(difference/np.maximum(np.abs(ref_state.astype(np.float64)),1.)))
            print('PHASE70 FINAL HIDDEN: mixed_failures=%d/896 max_abs=%.9g mean_abs=%.9g norm_max=%.9g'%
                  (violations,float(difference.max()),float(difference.mean()),norm),flush=True)
            print('PHASE70 computing independent reference next-token argmax (original INT4 output head)...',flush=True)
            ref_token,ref_logit,ref_second=head_reference(hm,ref_state,head_meta)
            print('PHASE70 REFERENCE TOKEN',ref_token,'logit=%.9g'%ref_logit,'margin=%.9g'%(ref_logit-ref_second),flush=True)
            print('PHASE70 DEVICE TOKEN',seen_token[0],flush=True)
            token_match=bool(ref_token==seen_token[0])
            print('PHASE70 TOKEN MATCH',int(token_match),flush=True)
            report={'prompt':args.prompt,'token_ids':ids,'device_token':seen_token[0],
                    'reference_token':ref_token,'token_match':token_match,'atol':args.atol,'rtol':args.rtol,
                    'positions_verified':2,'layers_per_position':24,'hidden_mixed_failures':violations,'hidden_max_abs':float(difference.max()),
                    'hidden_mean_abs':float(difference.mean()),'hidden_norm_max':norm,
                    'device_head_logit':seen_token[1],'reference_head_logit':ref_logit,
                    'comparison':'FP32 reference of genuine pretrained INT4 transformer image and INT4 head, not BF16 original Qwen'}
            Path(args.report).write_text(json.dumps(report,indent=2),encoding='utf-8')
            print('PHASE70 REPORT SAVED',args.report,flush=True)
            if violations:
                print('PHASE70 WARNING: final hidden state fails selected tolerance.',flush=True)
            if not token_match:
                print('PHASE70 FAIL: device token differs from independently computed INT4 stack.',flush=True)
            else:
                print('PHASE70 PASS: device argmax agrees with independent INT4 stack (full hidden tolerance reported separately).',flush=True)
            if args.strict and (violations or not token_match):
                raise SystemExit(2)
        finally:hm.close();mm.close()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image',default='phase25_qwen_24layers_int4.bin')
    p.add_argument('--metadata',default='phase25_qwen_24layers_int4.json')
    p.add_argument('--head',default='phase45_qwen_head_int4.bin')
    p.add_argument('--head-metadata',default='phase45_qwen_head_int4.json')
    p.add_argument('--port',default='COM3');p.add_argument('--timeout',type=int,default=7200)
    p.add_argument('--prompt',default='Hi?',help='Text that tokenizes to EXACTLY two Qwen tokens')
    p.add_argument('--tokenizer-file',default=None,help='Optional local tokenizer.json')
    p.add_argument('--checkpoint',default=None,help='Optional official model.safetensors path')
    p.add_argument('--prepare-only',action='store_true',help='Only tokenize and look up original pretrained embedding rows, no ESP32')
    p.add_argument('--active-layers',type=int,default=24)
    p.add_argument('--atol',type=float,default=0.001)
    p.add_argument('--rtol',type=float,default=0.001)
    p.add_argument('--capture',default='phase70_device_capture.npz')
    p.add_argument('--reference-capture',default='phase70_reference_capture.npz')
    p.add_argument('--report',default='phase70_two_token_report.json')
    p.add_argument('--strict',action='store_true')
    a=p.parse_args()
    if a.active_layers!=24:raise ValueError('Phase70 requires all 24 layers')
    if a.atol<=0 or a.rtol<0:raise ValueError('Invalid tolerance')
    if a.active_layers==24:
        print('WARNING: full 24-layer grouped-Q15 inference is not precision-validated; experimental output only.',flush=True)
    run(a)

if __name__=='__main__':main()
