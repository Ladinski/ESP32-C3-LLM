"""Phase 85: ESP32-C3 fixed-prompt prefill and one next-token prediction.

PC only streams immutable pretrained weights and token embeddings during inference.
Bounded causal GQA attention and KV cache execute on ESP32.
Offline reference checks happen only after device inference completes.
Requires Phase85 firmware and Phase80 export.
"""
import argparse
import hashlib
import json
import mmap
import struct
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / 'phase73_instruct' / 'phase80_int8_images' / 'qwen25_instruct_w8_manifest.json'
D, FF, KV = 896, 4864, 128
MATS = [('q_proj',D,D),('k_proj',KV,D),('v_proj',KV,D),('o_proj',D,D),('gate_proj',FF,D),('up_proj',FF,D),('down_proj',D,FF)]


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20), b''):h.update(b)
    return h.hexdigest()


def verify(manifest):
    meta=json.loads(Path(manifest).read_text(encoding='utf-8'))
    if meta.get('model')!='Qwen/Qwen2.5-0.5B-Instruct' or meta['transformer']['format']!='QW8S-v1' or len(meta['transformer']['layers'])!=24:
        raise ValueError('Unexpected INT8 model metadata')
    folder=Path(manifest).parent
    transformer=folder/'qwen25_instruct_transformer_w8.bin'
    head=folder/'qwen25_instruct_head_bf16.bin'
    for p,section in ((transformer,'transformer'),(head,'head')):
        if p.stat().st_size != meta[section]['bytes'] or sha256(p)!=meta[section]['sha256']:
            raise ValueError(f'Image integrity failed: {p}')
    if meta['head']['row_bytes']!=1792 or meta['head']['head_offset']!=3584:
        raise ValueError('Unexpected BF16 head format')
    return meta,transformer,head


def load_embedding(token_id):
    from huggingface_hub import hf_hub_download
    from phase73_instruct.export_qwen_layer0 import BFloatReader
    path=hf_hub_download('Qwen/Qwen2.5-0.5B-Instruct','model.safetensors',local_files_only=True)
    with BFloatReader(path) as r:
        result=np.asarray(r.get_row('model.embed_tokens.weight',int(token_id)),dtype='<f4')
    if result.shape != (D,):raise ValueError('Embedding length mismatch')
    return result


def mat_reference(mm, desc, x):
    """Pure NumPy reference, strictly AFTER device output has been collected."""
    rows,cols=desc['rows'],desc['cols']
    if x.shape!=(cols,):raise ValueError('Reference input shape mismatch')
    offset=desc['offset']; stride=desc['row_bytes']
    # Matrix records are [int8*cols][float32 scale] per row.
    raw=np.ndarray((rows,stride),dtype=np.uint8,buffer=mm,offset=offset)
    weights=raw[:,:cols].view(np.int8).astype(np.float32)
    scales=np.ndarray((rows,),dtype='<f4',buffer=mm,offset=offset+cols,strides=(stride,))
    return (weights@x.astype(np.float32))*scales


def rms(x,scale):
    return (x.astype(np.float32)/np.sqrt(np.mean(x*x,dtype=np.float32)+np.float32(1e-6)))*scale


def first_layer_reference(mm, layer, embed):
    """Uses all seven exported INT8 projections, including Q/K on the one-token path."""
    norm_in=np.frombuffer(mm,dtype='<f4',count=D,offset=layer['norms'][0]).copy()
    norm_post=np.frombuffer(mm,dtype='<f4',count=D,offset=layer['norms'][1]).copy()
    maps={m['name']:m for m in layer['matrices']}
    x=rms(embed,norm_in)
    q=mat_reference(mm,maps['q_proj'],x)+np.frombuffer(mm,dtype='<f4',count=D,offset=layer['biases'][0])
    k=mat_reference(mm,maps['k_proj'],x)+np.frombuffer(mm,dtype='<f4',count=KV,offset=layer['biases'][1])
    v=mat_reference(mm,maps['v_proj'],x)+np.frombuffer(mm,dtype='<f4',count=KV,offset=layer['biases'][2])
    if not np.isfinite(q).all() or not np.isfinite(k).all():raise RuntimeError('Bad Q or K')
    # Single-position causal attention, all heads attend exclusively to position zero.
    head_out=np.repeat(v.reshape(2,64),7,axis=0).reshape(D)
    x1=embed+mat_reference(mm,maps['o_proj'],head_out)
    h=rms(x1,norm_post)
    g=mat_reference(mm,maps['gate_proj'],h)
    u=mat_reference(mm,maps['up_proj'],h)
    act=(g/(1.0+np.exp(-np.clip(g,-80,80))))*u
    return x1+mat_reference(mm,maps['down_proj'],act.astype(np.float32))



def first_layer_position_reference(mm, layer, embeddings):
    """Offline NumPy reference for position 1 or 2 using full causal GQA and RoPE.

    Recompute layer-1 K/V from token embeddings only after the device finishes.
    This is independent of the device KV cache and is not used during inference.
    """
    norm_in=np.frombuffer(mm,dtype='<f4',count=D,offset=layer['norms'][0]).copy()
    norm_post=np.frombuffer(mm,dtype='<f4',count=D,offset=layer['norms'][1]).copy()
    maps={m['name']:m for m in layer['matrices']}
    def proj(name,x):return mat_reference(mm,maps[name],x)
    angles=np.power(1000000.0,-np.arange(0,64,2,dtype=np.float32)/64.0)
    def rope(t,pos):
        a=t.reshape(-1,64).copy()
        c=np.cos(angles*np.float32(pos)).astype(np.float32)
        sn=np.sin(angles*np.float32(pos)).astype(np.float32)
        left=a[:,:32].copy();right=a[:,32:].copy()
        a[:,:32]=left*c-right*sn;a[:,32:]=right*c+left*sn
        return a
    keys=[];values=[]
    q=None
    for pos,emb in enumerate(embeddings):
        norm=rms(emb,norm_in)
        k=proj('k_proj',norm)+np.frombuffer(mm,dtype='<f4',count=KV,offset=layer['biases'][1])
        v=proj('v_proj',norm)+np.frombuffer(mm,dtype='<f4',count=KV,offset=layer['biases'][2])
        keys.append(rope(k,pos).reshape(2,64))
        values.append(v.reshape(2,64))
        if pos==len(embeddings)-1:
            q=proj('q_proj',norm)+np.frombuffer(mm,dtype='<f4',count=D,offset=layer['biases'][0])
            q=rope(q,pos).reshape(14,64)
    kv_index=np.arange(14)//7
    scores=np.stack([np.sum(q*k[kv_index],axis=1,dtype=np.float32)*np.float32(0.125)
                     for k in keys],axis=1)
    scores=scores-np.max(scores,axis=1,keepdims=True)
    weights=np.exp(scores.astype(np.float32)).astype(np.float32)
    weights=weights/np.sum(weights,axis=1,keepdims=True)
    mixed=np.zeros((14,64),dtype=np.float32)
    for pos,v in enumerate(values):
        mixed+=weights[:,pos,None]*v[kv_index]
    emb=embeddings[-1]
    x1=emb+proj('o_proj',mixed.reshape(D))
    h=rms(x1,norm_post)
    g=proj('gate_proj',h);u=proj('up_proj',h)
    act=(g/(1+np.exp(-np.clip(g,-80,80))))*u
    return x1+proj('down_proj',act.astype(np.float32))

def head_reference(path, hidden):
    # BF16 tied embedding output-head rows, FP32 final RMSNorm.
    with open(path,'rb') as f,mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as hm:
        norm=np.frombuffer(hm,dtype='<f4',count=D,offset=0).copy()
        x=rms(hidden,norm)
        n=151936; start=3584; best=-np.inf; arg=-1
        for pos in range(0,n,1024):
            count=min(1024,n-pos)
            b=np.frombuffer(hm,dtype='<u2',count=count*D,offset=start+pos*D*2).copy()
            w=(b.astype(np.uint32)<<16).view(np.float32).reshape(count,D)
            vals=w@x
            i=int(np.argmax(vals))
            if float(vals[i])>best: best=float(vals[i]);arg=pos+i
    return arg,best



PROMPT = "Capital of France:"
MAX_PROMPT_TOKENS = 6


def tokenize_prompt():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct', local_files_only=True)
    ids = tokenizer.encode(PROMPT, add_special_tokens=False)
    if not 1 <= len(ids) <= MAX_PROMPT_TOKENS:
        raise ValueError(f'Prompt requires {len(ids)} tokens but device capacity is 1..{MAX_PROMPT_TOKENS}. No inference attempted.')
    if tokenizer.decode(ids,skip_special_tokens=False) != PROMPT:
        raise ValueError('Tokenizer round-trip failed')
    return tokenizer, ids


def run(args):
    import serial
    tokenizer,ids=tokenize_prompt()
    print('PHASE85 PROMPT',repr(PROMPT),flush=True)
    print('PHASE85 INPUT TOKEN IDS',ids,'count',len(ids),flush=True)
    print('PHASE85 INPUT TOKEN PIECES',[tokenizer.decode([i]) for i in ids],flush=True)
    if args.prepare_only:
        print('PHASE85 PREPARE PASS count',len(ids),'max',MAX_PROMPT_TOKENS,flush=True)
        return
    meta,image,head=verify(Path(args.manifest).resolve())
    print('PHASE85 VERIFIED IMAGES',meta['transformer']['sha256'],meta['head']['sha256'],flush=True)
    layer_by_id=meta['transformer']['layers']
    checkpoints={}
    embeddings=[load_embedding(i) for i in ids]
    pending=None
    selected=None
    head_pos=len(ids)-1
    start=time.monotonic()
    completed=False
    with open(image,'rb') as f,mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as mm,open(head,'rb') as hf,mmap.mmap(hf.fileno(),0,access=mmap.ACCESS_READ) as hm:
        with serial.Serial(args.port,115200,timeout=2,write_timeout=30) as dev:
            time.sleep(0.6)
            dev.reset_input_buffer()
            dev.write(b'PING\n')
            expiry=time.monotonic()+15
            while time.monotonic()<expiry:
                if dev.readline().decode('ascii','replace').strip()=='PONG P85':break
            else:raise TimeoutError('No PONG P85. Flash Phase85 firmware; close serial monitor.')
            dev.write(f'STARTP {len(ids)}\n'.encode('ascii'))
            while time.monotonic()-start<args.timeout:
                line=dev.readline().decode('ascii','replace').strip()
                if not line:continue
                parts=line.split()
                if parts[0]=='EMBED_POS':
                    if len(parts)!=3 or parts[2]!='3584':raise RuntimeError(line)
                    pos=int(parts[1])
                    if pos<0 or pos>=len(ids):raise RuntimeError(line)
                    if pos>0 and sum(1 for pp,ll in checkpoints if pp==pos-1)!=24:raise RuntimeError('Previous position incomplete')
                    dev.write(embeddings[pos].astype('<f4').tobytes())
                    print(f'PHASE85 PREFILL position={pos+1}/{len(ids)} token_id={ids[pos]}',flush=True)
                elif parts[0]=='PARAM':
                    if len(parts)!=5:raise RuntimeError(line)
                    li,typ,name,n=int(parts[1]),parts[2],parts[3],int(parts[4])
                    if not 0<=li<24:raise RuntimeError(line)
                    layer=layer_by_id[li]
                    if typ=='norm':
                        ix={'input_layernorm':0,'post_attention_layernorm':1}.get(name)
                        if ix is None or n!=3584:raise RuntimeError(line)
                        off=layer['norms'][ix]
                    elif typ=='bias':
                        ix={'q_proj':0,'k_proj':1,'v_proj':2}.get(name)
                        if ix is None or n!=(3584 if ix==0 else 512):raise RuntimeError(line)
                        off=layer['biases'][ix]
                    else:raise RuntimeError(line)
                    dev.write(mm[off:off+n])
                elif parts[0]=='MATRIX':
                    if pending is not None and pending[2]!=pending[3]:raise RuntimeError('Previous weight transfer incomplete')
                    if len(parts)!=5:raise RuntimeError(line)
                    li,name,rows,cols=int(parts[1]),parts[2],int(parts[3]),int(parts[4])
                    if not 0<=li<24:raise RuntimeError(line)
                    matches=[m for m in layer_by_id[li]['matrices'] if m['name']==name]
                    if len(matches)!=1:raise RuntimeError(line)
                    m=matches[0]
                    if m['rows']!=rows or m['cols']!=cols or m['row_bytes']!=cols+4:raise RuntimeError(line)
                    off=m['offset'];total=rows*(cols+4)
                    pending=[mm,off,0,total]
                    print(f'PHASE85 MATRIX layer={li+1} {name} rows={rows} cols={cols}',flush=True)
                elif parts[0]=='CHUNK':
                    if len(parts)!=2 or pending is None:raise RuntimeError('Unsolicited CHUNK: '+line)
                    n=int(parts[1])
                    if n<=0 or n>7168 or pending[2]+n>pending[3]:raise RuntimeError('Invalid CHUNK: '+line)
                    source,base,offset,total=pending
                    for start_byte in range(offset,offset+n,512):
                        block=source[base+start_byte:base+min(start_byte+512,offset+n)]
                        if dev.write(block)!=len(block):raise RuntimeError('Partial serial chunk write')
                    pending[2]+=n
                elif parts[0]=='STATE':
                    if pending is not None and pending[2]!=pending[3]:raise RuntimeError('Previous weight transfer incomplete before STATE')
                    pending=None
                    if len(parts)!=3:raise RuntimeError(line)
                    li,bytes_count=map(int,parts[1:])
                    if bytes_count!=3584:raise RuntimeError(line)
                    # Determine active position based on saved checkpoints.
                    pos=len(checkpoints)//24
                    if pos>=len(ids) or li!=sum(1 for (p,_) in checkpoints if p==pos):raise RuntimeError(line)
                    dev.write(b'SEND\n')
                    blob=dev.read(bytes_count)
                    if len(blob)!=bytes_count:raise RuntimeError('Truncated STATE')
                    checkpoints[(pos,li)]=np.frombuffer(blob,dtype='<f4').copy()
                    if not np.isfinite(checkpoints[(pos,li)]).all():raise RuntimeError('Nonfinite state')
                    dev.write(b'GOT\n')
                elif parts[0]=='LAYER_DONE':
                    if len(parts)!=3:raise RuntimeError(line)
                    pidx,li=int(parts[1]),int(parts[2])
                    if pidx>=len(ids) or li!=sum(1 for (p,_) in checkpoints if p==pidx):raise RuntimeError('Missing layer state')
                    if li==24:print(f'PHASE85 PREFILL position {pidx+1}/{len(ids)} all 24 layers completed',flush=True)
                elif parts[0]=='HEAD_NORM':
                    if parts!=['HEAD_NORM','3584'] or len(checkpoints)!=len(ids)*24:raise RuntimeError(line)
                    dev.write(hm[:3584])
                elif parts[0]=='HEAD_ROWS':
                    if parts!=['HEAD_ROWS','151936','1792'] or len(checkpoints)!=len(ids)*24:raise RuntimeError(line)
                    if pending is not None and pending[2]!=pending[3]:raise RuntimeError('Previous transfer incomplete')
                    pending=[hm,3584,0,len(hm)-3584]
                    print('PHASE85 OUTPUT HEAD device-driven BF16 streaming',flush=True)
                elif parts[0]=='TOKEN':
                    if pending is None or pending[2]!=pending[3]:raise RuntimeError('Head transfer incomplete')
                    pending=None
                    if len(parts)!=3 or selected is not None:raise RuntimeError(line)
                    selected=(int(parts[1]),float(parts[2]))
                    print('PHASE85 ESP32 SELECTED',selected,flush=True)
                elif parts[0]=='DONE':
                    if pending is not None and pending[2]!=pending[3]:raise RuntimeError('Transfer incomplete at DONE')
                    if len(parts)!=5 or parts[1]!='P85' or parts[2]!=f'positions={len(ids)}':raise RuntimeError(line)
                    print(line,flush=True)
                    completed=True
                    break
                elif parts[0]=='ERR':raise RuntimeError('DEVICE '+line)
                else:raise RuntimeError('Unexpected device line: '+line)
            else:raise TimeoutError('Device inference timed out')
        if not completed or len(checkpoints)!=len(ids)*24 or selected is None:raise RuntimeError('Missing prompt outputs')
        print('PHASE85 ESP32 finished; independent checks begin',flush=True)
        validation=[]
        for pos in range(len(ids)):
            ref=(first_layer_reference(mm,layer_by_id[0],embeddings[0]) if pos==0
                 else first_layer_position_reference(mm,layer_by_id[0],embeddings[:pos+1]))
            actual=checkpoints[(pos,0)]
            valid=np.isclose(ref,actual,atol=args.atol,rtol=args.rtol)
            fail=int(np.count_nonzero(~valid))
            err=float(np.max(np.abs(ref.astype(np.float64)-actual.astype(np.float64))))
            print(f'PHASE85 POSITION{pos+1} LAYER1 CHECK failures {fail} / {D} max_abs {err}',flush=True)
            validation.append({'position':pos,'failures':fail,'max_abs':err})
            if fail:raise RuntimeError(f'PHASE85 LAYER1 VALIDATION FAILED AT POSITION {pos+1}')
        ref_id,ref_score=head_reference(head,checkpoints[(head_pos,23)])
        score_diff=abs(ref_score-selected[1])
        print('PHASE85 HEAD AUDIT device',selected[0],'independent_same_hidden',ref_id,'score_difference',score_diff,flush=True)
        if selected[0]!=ref_id or score_diff>0.1:raise RuntimeError('PHASE85 HEAD AUDIT FAILED')
        out=tokenizer.decode([selected[0]],skip_special_tokens=False)
        full=tokenizer.decode(ids+[selected[0]],skip_special_tokens=False)
        print('PHASE85 MODEL OUTPUT TOKEN',repr(out),flush=True)
        print('PHASE85 COMPLETED TEXT',repr(full),flush=True)
        target_ok=(out.strip().lower()=='paris')
        print('PHASE85 PARIS TARGET', 'MATCH' if target_ok else 'NOT MATCHED (model output preserved)',flush=True)
        print(f'PHASE85 DEVICE GATE PASS prompt_tokens={len(ids)} generated_tokens=1 layers=24',flush=True)
    Path(args.report).write_text(json.dumps({'prompt':PROMPT,'prompt_token_ids':ids,'generated_token':selected[0],
        'generated_piece':out,'full_text':full,'target_paris':target_ok,'head_score_difference':score_diff,
        'layer1_checks':validation,'elapsed_seconds':time.monotonic()-start},indent=2),encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',default=str(DEFAULT_MANIFEST))
    p.add_argument('--port',default='COM3')
    p.add_argument('--timeout',type=int,default=14400)
    p.add_argument('--atol',type=float,default=0.003)
    p.add_argument('--rtol',type=float,default=0.001)
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--report',default='phase85_paris_prompt_report.json')
    run(p.parse_args())
if __name__=='__main__':main()
