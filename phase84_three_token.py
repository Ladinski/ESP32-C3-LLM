"""Phase 84: ESP32-C3 pretrained INT8 three-position autoregressive generation.

PC only streams immutable pretrained weights and token embeddings during inference.
Three-position causal GQA attention and KV cache execute on ESP32.
Offline reference checks happen only after device inference completes.
Requires Phase84 firmware and Phase80 export.
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


def run(args):
    import serial
    meta,image,head=verify(Path(args.manifest).resolve())
    print('PHASE84 VERIFIED IMAGES',meta['transformer']['sha256'],meta['head']['sha256'],flush=True)
    embed=load_embedding(args.token_id)
    if args.prepare_only:
        print('PHASE84 PREPARE PASS: token',args.token_id,'positions 3 layers 24',flush=True)
        return
    layer_by_id=meta['transformer']['layers']
    checkpoints={}
    tokens=[]
    pos=0
    embeddings=[embed]
    pending=None  # [mmap, base offset, sent, total] for device-driven chunks
    start=time.monotonic()
    with open(image,'rb') as f,mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as mm,open(head,'rb') as hf,mmap.mmap(hf.fileno(),0,access=mmap.ACCESS_READ) as hm:
        with serial.Serial(args.port,115200,timeout=2,write_timeout=30) as dev:
            time.sleep(0.6)
            dev.reset_input_buffer()
            dev.write(b'PING\n')
            expiry=time.monotonic()+15
            while time.monotonic()<expiry:
                if dev.readline().decode('ascii','replace').strip()=='PONG P84':break
            else:raise TimeoutError('No PONG P84. Flash Phase84 firmware, close serial monitor.')
            dev.write(b'START3\n')
            while time.monotonic()-start<args.timeout:
                line=dev.readline().decode('ascii','replace').strip()
                if not line:continue
                parts=line.split()
                if parts[0]=='EMBED':
                    if parts!=['EMBED','3584']:raise RuntimeError(line)
                    dev.write(embed.astype('<f4').tobytes())
                elif parts[0]=='EMBED_TOKEN':
                    if len(parts)!=3 or len(tokens)!=pos+1 or pos>=2 or int(parts[1])!=tokens[-1][0] or parts[2]!='3584':raise RuntimeError(line)
                    pos+=1
                    next_embed=load_embedding(tokens[-1][0])
                    embeddings.append(next_embed)
                    dev.write(next_embed.astype('<f4').tobytes())
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
                    if pending is not None and pending[2]!=pending[3]:
                        raise RuntimeError('Previous weight transfer incomplete')
                    if len(parts)!=5:raise RuntimeError(line)
                    li,name,rows,cols=int(parts[1]),parts[2],int(parts[3]),int(parts[4])
                    if not 0<=li<24:raise RuntimeError(line)
                    matches=[m for m in layer_by_id[li]['matrices'] if m['name']==name]
                    if len(matches)!=1:raise RuntimeError(line)
                    m=matches[0]
                    if m['rows']!=rows or m['cols']!=cols or m['row_bytes']!=cols+4:raise RuntimeError(line)
                    off=m['offset'];total=rows*(cols+4)
                    pending=[mm,off,0,total]
                    print(f'PHASE84 MATRIX position={pos+1} layer={li+1} {name} rows={rows} cols={cols}',flush=True)
                elif parts[0]=='CHUNK':
                    if len(parts)!=2 or pending is None:raise RuntimeError('Unsolicited CHUNK: '+line)
                    n=int(parts[1])
                    if n<=0 or n>7168 or pending[2]+n>pending[3]:
                        raise RuntimeError('Invalid CHUNK: '+line)
                    source,base,offset,total=pending
                    for start_byte in range(offset,offset+n,512):
                        block=source[base+start_byte:base+min(start_byte+512,offset+n)]
                        if dev.write(block)!=len(block):raise RuntimeError('Partial serial chunk write')
                    pending[2]+=n
                elif parts[0]=='STATE':
                    if pending is not None and pending[2]!=pending[3]:
                        raise RuntimeError('Previous weight transfer incomplete before STATE')
                    pending=None
                    if len(parts)!=3:raise RuntimeError(line)
                    li,bytes_count=map(int,parts[1:])
                    if bytes_count!=3584 or li!=sum(1 for (p,_) in checkpoints if p==pos):raise RuntimeError(line)
                    dev.write(b'SEND\n')
                    blob=dev.read(bytes_count)
                    if len(blob)!=bytes_count:raise RuntimeError('Truncated STATE')
                    checkpoints[(pos,li)]=np.frombuffer(blob,dtype='<f4').copy()
                    if not np.isfinite(checkpoints[(pos,li)]).all():raise RuntimeError('Nonfinite state')
                    dev.write(b'GOT\n')
                elif parts[0]=='LAYER_DONE':
                    if len(parts)!=3:raise RuntimeError(line)
                    pidx,li=int(parts[1]),int(parts[2])
                    if pidx!=pos or li!=sum(1 for (pp,_) in checkpoints if pp==pos):raise RuntimeError('Missing layer state')
                    print(f'ESP32 position {pos+1}/3 layer {li}/24 completed',flush=True)
                elif parts[0]=='HEAD_NORM':
                    if parts!=['HEAD_NORM','3584'] or sum(1 for (pp,_) in checkpoints if pp==pos)!=24:raise RuntimeError(line)
                    dev.write(hm[:3584])
                elif parts[0]=='HEAD_ROWS':
                    if parts!=['HEAD_ROWS','151936','1792'] or sum(1 for (pp,_) in checkpoints if pp==pos)!=24:raise RuntimeError(line)
                    if pending is not None and pending[2]!=pending[3]:raise RuntimeError('Previous transfer incomplete')
                    pending=[hm,3584,0,len(hm)-3584]
                    print('ESP32 output head: device-driven BF16 streaming',flush=True)
                elif parts[0]=='TOKEN':
                    if pending is None or pending[2]!=pending[3]:raise RuntimeError('Head transfer incomplete')
                    pending=None
                    if len(parts)!=3:raise RuntimeError(line)
                    tokens.append((int(parts[1]),float(parts[2])))
                    if len(tokens)!=pos+1:raise RuntimeError('Token position mismatch')
                    print('ESP32 SELECTED TOKEN position',pos, tokens[-1],flush=True)
                elif parts[0]=='DONE':
                    if pending is not None and pending[2]!=pending[3]:raise RuntimeError('Transfer incomplete at DONE')
                    print(line,flush=True)
                    break
                elif parts[0]=='ERR':raise RuntimeError('DEVICE '+line)
                else:raise RuntimeError('Unexpected device line: '+line)
            else:raise TimeoutError('Device inference timed out')
        print('ESP32 inference ended; beginning independent offline checks',flush=True)
        if len(checkpoints)!=72 or len(tokens)!=3 or len(embeddings)!=3:raise RuntimeError('Missing three-position outputs')
        first=first_layer_reference(mm,layer_by_id[0],embed)
        d=np.abs(first.astype(np.float64)-checkpoints[(0,0)].astype(np.float64))
        ok=np.isclose(first,checkpoints[(0,0)],atol=args.atol,rtol=args.rtol)
        bad=int(np.count_nonzero(~ok))
        print('PHASE84 POSITION1 LAYER1 CHECK failures',bad,'/',D,'max_abs',float(np.max(d)),flush=True)
        if bad:raise RuntimeError('PHASE84 POSITION1 LAYER1 FAIL')
        if tokens[0][0]!=11:raise RuntimeError('Phase82 regression: first token changed from expected 11')
        for position_idx in range(3):
            ref_id,ref_score=head_reference(head,checkpoints[(position_idx,23)])
            token=tokens[position_idx]
            print('PHASE84 HEAD AUDIT position',position_idx,'device',token[0],
                  'independent_same_hidden',ref_id,'score_difference',abs(ref_score-token[1]),flush=True)
            if token[0]!=ref_id or abs(ref_score-token[1])>0.1:raise RuntimeError('PHASE84 HEAD AUDIT FAIL')
        later_failures={}
        for idx in (1,2):
            reference=first_layer_position_reference(mm,layer_by_id[0],embeddings[:idx+1])
            err=np.abs(reference.astype(np.float64)-checkpoints[(idx,0)].astype(np.float64))
            valid=np.isclose(reference,checkpoints[(idx,0)],atol=args.atol,rtol=args.rtol)
            bad_n=int(np.count_nonzero(~valid))
            later_failures[idx]=bad_n
            print(f'PHASE84 POSITION{idx+1} LAYER1 CHECK failures',bad_n,'/',D,'max_abs',float(np.max(err)),flush=True)
            if bad_n:raise RuntimeError(f'PHASE84 POSITION{idx+1} ATTENTION CHECK FAIL')
        print('PHASE84 DEVICE GATE PASS positions=3 layers=24',flush=True)
    Path(args.report).write_text(json.dumps({'positions':3,'layers':24,'initial_token_id':args.token_id,
        'first_layer_failures':bad,'later_position_layer1_failures':later_failures,
        'tokens':tokens,'elapsed_seconds':time.monotonic()-start},indent=2),encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',default=str(DEFAULT_MANIFEST))
    p.add_argument('--port',default='COM3')
    p.add_argument('--token-id',type=int,default=9707,help='Default Hello, one pretrained token only')
    # Phase84 always executes two positions with all 24 transformer layers
    p.add_argument('--timeout',type=int,default=14400)
    p.add_argument('--atol',type=float,default=0.003)
    p.add_argument('--rtol',type=float,default=0.001)
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--report',default='phase84_three_token_report.json')
    run(p.parse_args())
if __name__=='__main__':main()
