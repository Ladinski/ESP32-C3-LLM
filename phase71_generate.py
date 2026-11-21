"""Phase 71: two greedy tokens from ONE input token on ESP32-C3.
PC is an immutable weight/head/embedding byte server, not a neural executor.
The ESP32 selects token 0, which is fed back via its pretrained embedding
for token 1. All 24 transformer layers, causal KV, logits and argmax are on ESP32.
"""
import argparse
import json
import math
import mmap
import time
from pathlib import Path
from phase71_support import verify_image, prepare_prompt, head_reference, D, PAGE


def run(args):
    import numpy as np
    import serial
    from export_phase45_head import BFloatReader, REPO
    from huggingface_hub import hf_hub_download

    tokenizer, ids, embeddings = prepare_prompt(args)
    if len(ids) != 1:
        raise ValueError(f'Phase71 requires ONE prompt token, got {ids}. Use --prompt "Hello".')
    if args.prepare_only:
        print('PHASE71 PREPARE PASS: one prompt token, no ESP32 connection')
        return
    model_meta = json.loads(Path(args.metadata).read_text(encoding='utf-8'))
    head_meta = json.loads(Path(args.head_metadata).read_text(encoding='utf-8'))
    if len(model_meta['layers']) != 24 or head_meta['vocab'] != 151936 or head_meta['hidden'] != D:
        raise RuntimeError('Unexpected model metadata')
    if head_meta['row_bytes'] != D//2+4 or head_meta['head_offset'] != D*4:
        raise RuntimeError('Unexpected head layout')
    print('MODEL VERIFIED',verify_image(args.image,model_meta),flush=True)
    print('HEAD VERIFIED',verify_image(args.head,head_meta),flush=True)
    checkpoint = args.checkpoint or hf_hub_download(REPO,'model.safetensors')
    chosen = []
    reported_steps = []
    states = {}
    head_pending = False
    previous_step_verified = False
    finished = set()
    layer_seen = {0:set(),1:set()}
    correction = None
    done = False
    started = time.monotonic()
    output_head_count=0
    allowed = {(int(m['offset']),int(m['rows'])*int(m['row_bytes'])) for layer in model_meta['layers'] for m in layer['matrices']}

    with open(args.image,'rb') as model_file, open(args.head,'rb') as head_file, BFloatReader(checkpoint) as checkpoint_reader:
        mm=mmap.mmap(model_file.fileno(),0,access=mmap.ACCESS_READ)
        hm=mmap.mmap(head_file.fileno(),0,access=mmap.ACCESS_READ)
        try:
            with serial.Serial(args.port,115200,timeout=0.4,write_timeout=6) as dev:
                time.sleep(0.8)
                dev.write(b'START P71 GENERATE 2 24\n')
                print('SENT START P71 GENERATE 2 24',flush=True)
                deadline=time.monotonic()+args.timeout
                ready=False
                while time.monotonic()<deadline:
                    raw=dev.readline()
                    if not raw:
                        continue
                    line=raw.decode('ascii',errors='replace').strip()
                    if line=='READY PHASE71':
                        ready=True
                        print('READY PHASE71',flush=True)
                        continue
                    if not ready:
                        continue
                    if line.startswith('ERR '):
                        raise RuntimeError('ESP32: '+line)
                    if line.startswith('P46_EMBED '):
                        f=line.split()
                        if len(f)!=3:
                            raise RuntimeError('Bad P46_EMBED request: '+line)
                        pos,n=map(int,f[1:])
                        if pos not in (0,1) or n!=D*4:
                            raise RuntimeError('Invalid embedding request: '+line)
                        if pos==0:
                            v=embeddings[0]
                            id_to_embed=ids[0]
                        else:
                            if len(reported_steps)!=1 or len(chosen)!=1 or not previous_step_verified:
                                raise RuntimeError('ESP32 requested generated embedding without confirmed first-token choice')
                            id_to_embed=chosen[0][0]
                            v=np.asarray(checkpoint_reader.get_row('model.embed_tokens.weight',id_to_embed),dtype='<f4').tobytes()
                        if len(v)!=n or dev.write(v)!=n:
                            raise RuntimeError('Embedding write failure')
                        print(f'ESP32 embedding pos={pos} pretrained token={id_to_embed}',flush=True)
                    elif line.startswith('PAGE '):
                        f=line.split()
                        if len(f)!=3:raise RuntimeError('Bad PAGE request')
                        off,n=map(int,f[1:])
                        if off<0 or off>=len(mm) or off%PAGE or n!=min(PAGE,len(mm)-off):
                            raise RuntimeError('Invalid PAGE request: '+line)
                        block=mm[off:off+n]
                        if dev.write(block)!=len(block):raise RuntimeError('PAGE write failure')
                    elif line.startswith('RANGE '):
                        f=line.split()
                        if len(f)!=3:raise RuntimeError('Bad RANGE request')
                        off,n=map(int,f[1:])
                        if (off,n) not in allowed:raise RuntimeError('Unknown matrix RANGE request: '+line)
                        for at in range(0,n,PAGE):
                            block=mm[off+at:off+min(n,at+PAGE)]
                            if dev.write(block)!=len(block):raise RuntimeError('RANGE write failure')
                    elif line.startswith('LAYER '):
                        f=line.split()
                        if len(f)<3:raise RuntimeError('Malformed layer status')
                        pos,layer=int(f[1]),int(f[2])
                        if pos not in (0,1) or layer not in range(24):raise RuntimeError('Unexpected layer')
                        layer_seen[pos].add(layer)
                        print(f'  ESP32 position={pos} layer={layer+1}/24',flush=True)
                    elif line.startswith('TOKEN ') and line.endswith(' COMPLETE'):
                        f=line.split()
                        pos=int(f[1])
                        if pos not in (0,1) or layer_seen[pos]!=set(range(24)):
                            raise RuntimeError('Incomplete token inference: '+line)
                        if pos in finished:raise RuntimeError('Duplicate token complete')
                        finished.add(pos)
                        print(line,flush=True)
                    elif line.startswith('OUT '):
                        f=line.split()
                        if len(f)!=19:raise RuntimeError('Invalid OUT data length')
                        pos,start=map(int,f[1:3])
                        values=[float(t) for t in f[3:]]
                        if pos not in (0,1) or start not in range(0,D,16) or not all(math.isfinite(t) for t in values):
                            raise RuntimeError('Invalid OUT tensor')
                        if pos not in finished:raise RuntimeError('OUT before TOKEN COMPLETE')
                        chunks=states.setdefault(pos,{})
                        if start in chunks:raise RuntimeError('Duplicate OUT chunk')
                        chunks[start]=values
                    elif line.startswith('HEAD_NORM '):
                        n=int(line.split()[1])
                        if n!=D*4 or len(states.get(len(chosen),{}))!=D//16:
                            raise RuntimeError('Head started before complete device state')
                        norm=hm[:n]
                        if dev.write(norm)!=len(norm):raise RuntimeError('HEAD_NORM write failure')
                    elif line.startswith('HEAD_RANGE '):
                        f=line.split();off,n=map(int,f[1:])
                        if len(f)!=3 or off!=head_meta['head_offset'] or n!=head_meta['vocab']*head_meta['row_bytes']:
                            raise RuntimeError('Invalid head range')
                        output_head_count+=1
                        print(f'ESP32 output head {output_head_count}/2: streaming {n:,} INT4 bytes',flush=True)
                        for at in range(0,n,PAGE):
                            block=hm[off+at:off+min(n,at+PAGE)]
                            if dev.write(block)!=len(block):raise RuntimeError('HEAD_RANGE write failure')
                    elif line.startswith('P45_TOKEN '):
                        f=line.split()
                        if len(f)!=3:raise RuntimeError('Invalid token response')
                        token=int(f[1]);logit=float(f[2])
                        if not (0<=token<head_meta['vocab']) or not math.isfinite(logit):raise RuntimeError('Invalid token/logit')
                        if len(chosen)>=2 or output_head_count!=len(chosen)+1:
                            raise RuntimeError('Unexpected head selection sequence')
                        chosen.append((token,logit))
                        print(f'ESP32 GENERATED {len(chosen)}/2: token={token}, text={tokenizer.decode([token],skip_special_tokens=False)!r}, logit={logit:.7g}',flush=True)
                    elif line.startswith('PHASE71_STEP '):
                        f=line.split()
                        if len(f)!=3:raise RuntimeError('Bad P71_STEP')
                        pos,token=map(int,f[1:])
                        if pos!=len(reported_steps) or len(chosen)!=pos+1 or token!=chosen[pos][0]:
                            raise RuntimeError('Device step token differs from selected token')
                        if pos not in finished or len(states.get(pos,{}))!=D//16:
                            raise RuntimeError('Incomplete device state for selected token')
                        reported_steps.append(token)
                        previous_step_verified=True
                        print('PHASE71_STEP confirmed',pos,token,flush=True)
                    elif line.startswith('PHASE71_APPLIED '):
                        f=line.split()
                        if len(f)!=2 or f[1]!='rows=1792':
                            raise RuntimeError('Missing layer-3 correction proof')
                        correction=1792
                        print(line,flush=True)
                    elif line=='DONE PHASE71 GENERATE':
                        if len(chosen)!=2 or len(reported_steps)!=2 or finished!={0,1} or correction!=1792:
                            raise RuntimeError('Incomplete Phase71 generation proof')
                        done=True
                        print(line,flush=True)
                        break
                if not done:raise TimeoutError('Phase71 timed out; no successful completion')
            print('Computing independent output-head checks AFTER ESP32 finished...',flush=True)
            for pos in (0,1):
                hidden=[v for start in range(0,D,16) for v in states[pos][start]]
                reference_id,reference_logit,_=head_reference(hm,hidden,head_meta)
                token,logit=chosen[pos]
                difference=abs(reference_logit-logit)
                print(f'HEAD AUDIT {pos}: device={token} reference={reference_id} score_error={difference:.7g}',flush=True)
                if reference_id!=token or difference>0.05:
                    raise RuntimeError(f'HEAD AUDIT FAIL pos {pos}')
            output_ids=[t for t,_ in chosen]
            print('GENERATED IDS',output_ids,flush=True)
            print('GENERATED TEXT',repr(tokenizer.decode(output_ids,skip_special_tokens=False)),flush=True)
            print('TOTAL DEVICE SESSION SECONDS',round(time.monotonic()-started,2),flush=True)
            print('PHASE71 PASS: two ESP32-selected tokens; first token causally fed back via pretrained embedding; 24 layers each; both heads matched device-state NumPy.',flush=True)
            print('LIMIT: no independent full-transformer replay for these tokens; two-position KV limit; language quality unverified.',flush=True)
        finally:
            hm.close();mm.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prompt',default='Hello',help='Must tokenize to one Qwen token')
    p.add_argument('--port',default='COM3')
    p.add_argument('--timeout',type=int,default=10800)
    p.add_argument('--image',default='phase25_qwen_24layers_int4.bin')
    p.add_argument('--metadata',default='phase25_qwen_24layers_int4.json')
    p.add_argument('--head',default='phase45_qwen_head_int4.bin')
    p.add_argument('--head-metadata',default='phase45_qwen_head_int4.json')
    p.add_argument('--tokenizer-file',default=None)
    p.add_argument('--checkpoint',default=None)
    p.add_argument('--prepare-only',action='store_true')
    run(p.parse_args())

if __name__=='__main__':main()
