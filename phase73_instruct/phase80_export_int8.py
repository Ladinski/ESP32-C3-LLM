"""Phase 80: Export Qwen2.5-0.5B-Instruct W8A32 weights for later ESP32-C3 streaming.

Creates separate images; DOES NOT flash or perform device inference.
On-disk layout: 128-byte header; per-layer norms, Q/K/V biases (FP32);
7 linear matrices, each row [signed int8 cols][float32 little endian scale].
Head image: final RMSNorm FP32 followed by original BF16 head rows.
"""
import argparse
import hashlib
import json
import struct
from pathlib import Path
import numpy as np

D=896; KV=128; FF=4864; VOCAB=151936; PAGE=512
REPO='Qwen/Qwen2.5-0.5B-Instruct'
MATS=[('q_proj',D,D),('k_proj',KV,D),('v_proj',KV,D),('o_proj',D,D),('gate_proj',FF,D),('up_proj',FF,D),('down_proj',D,FF)]


def sha256(path):
    with open(path,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def get_checkpoint(explicit):
    if explicit:
        p=Path(explicit)
        if not p.is_file(): raise FileNotFoundError(p)
        return p
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(REPO, 'model.safetensors'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=str)
    p.add_argument('--output-dir',default='phase80_int8_images')
    p.add_argument('--layers',type=int,default=24)
    p.add_argument('--prompt',default='Hello')
    p.add_argument('--verify-only',action='store_true')
    args=p.parse_args()
    if not 1 <= args.layers <= 24: raise ValueError('layers must be 1..24')
    from export_qwen_layer0 import BFloatReader
    dest=Path(args.output_dir);dest.mkdir(parents=True,exist_ok=True)
    checkpoint=get_checkpoint(args.checkpoint)
    print('SOURCE CHECKPOINT',checkpoint,flush=True)
    image=dest/'qwen25_instruct_transformer_w8.bin'
    head=dest/'qwen25_instruct_head_bf16.bin'
    manifest_path=dest/'qwen25_instruct_w8_manifest.json'
    if args.verify_only:
        if not manifest_path.exists():raise FileNotFoundError(manifest_path)
        m=json.loads(manifest_path.read_text(encoding='utf-8'))
        for file, info in [(image,m['transformer']),(head,m['head'])]:
            if file.stat().st_size != info['bytes']:raise RuntimeError(f'{file} size differs')
            if sha256(file)!=info['sha256']:raise RuntimeError(f'{file} SHA mismatch')
        with BFloatReader(checkpoint) as reader, image.open('rb') as fh:
            for li in sorted(set([0,min(2,args.layers-1),args.layers-1])):
                layer=m['transformer']['layers'][li]
                for rec in layer['matrices']:
                    col=rec['cols'];rs=rec['rows'];stride=rec['row_bytes']
                    key=f'model.layers.{li}.'+('self_attn.' if rec['name'] in ('q_proj','k_proj','v_proj','o_proj') else 'mlp.')+rec['name']+'.weight'
                    source=reader.get_tensor(key)
                    for idx in sorted(set([0, rs//2, rs-1])):
                        fh.seek(rec['offset']+idx*stride)
                        q=np.frombuffer(fh.read(col),dtype=np.int8).astype(np.float32)
                        sc=struct.unpack('<f',fh.read(4))[0]
                        if not np.isfinite(sc) or sc<=0:raise RuntimeError('Invalid scale')
                        original=source[idx]
                        expected=np.clip(np.rint(original/sc),-127,127).astype(np.int8)
                        if not np.array_equal(q.astype(np.int8),expected):raise RuntimeError(f'INT8 row mismatch {key} row {idx}')
                print(f'RECONSTRUCTION VERIFIED layer={li+1} all seven matrices, 3 rows each',flush=True)
        print('PHASE80 IMAGE VERIFY PASS',flush=True)
        return
    if image.exists() or head.exists() or manifest_path.exists():
        raise FileExistsError('Phase80 output exists; pass --verify-only or choose a new --output-dir (no overwrite)')
    meta={'model':REPO,'transformer':{'format':'QW8S-v1','layers':[],'quantization':'signed symmetric INT8 per output row; FP32 scales; FP32 norm and biases'},'head':{'format':'FP32 RMSNorm then original BF16 tied embeddings','vocab':VOCAB,'hidden':D}}
    with BFloatReader(checkpoint) as sf, image.open('wb') as out:
        out.write(bytes(128))
        for li in range(args.layers):
            prefix=f'model.layers.{li}.'
            info={'index':li,'norms':[],'biases':[],'matrices':[]}
            for key in ('input_layernorm.weight','post_attention_layernorm.weight'):
                info['norms'].append(out.tell())
                out.write(np.asarray(sf.get_tensor(prefix+key),dtype='<f4').tobytes())
            for name in ('q_proj','k_proj','v_proj'):
                info['biases'].append(out.tell())
                out.write(np.asarray(sf.get_tensor(prefix+'self_attn.'+name+'.bias'),dtype='<f4').tobytes())
            for name,rows,cols in MATS:
                key=prefix+('self_attn.' if name in ('q_proj','k_proj','v_proj','o_proj') else 'mlp.')+name+'.weight'
                arr=sf.get_tensor(key)
                if arr.shape!=(rows,cols):raise ValueError(f'Unexpected {key} shape: {arr.shape}')
                offset=out.tell()
                for row in arr:
                    peak=float(np.max(np.abs(row)))
                    scale=np.float32(peak/127. if peak else 1.)
                    q=np.clip(np.rint(row/scale),-127,127).astype(np.int8)
                    out.write(q.tobytes())
                    out.write(struct.pack('<f',float(scale)))
                info['matrices'].append({'name':name,'offset':offset,'rows':rows,'cols':cols,'row_bytes':cols+4})
            meta['transformer']['layers'].append(info)
            print(f'EXPORTED INT8 LAYER {li+1}/{args.layers}, {out.tell():,} bytes',flush=True)
        size=out.tell()
        out.seek(0)
        out.write(struct.pack('<4s7I',b'QW8S',1,D,KV,FF,PAGE,args.layers,size))
    meta['transformer']['bytes']=image.stat().st_size
    meta['transformer']['sha256']=sha256(image)
    with BFloatReader(checkpoint) as sf,head.open('wb') as out:
        norm=np.asarray(sf.get_tensor('model.norm.weight'),dtype='<f4')
        if norm.shape!=(D,):raise ValueError('Unexpected final norm dimensions')
        out.write(norm.tobytes())
        name='lm_head.weight' if 'lm_head.weight' in sf.meta else 'model.embed_tokens.weight'
        tensor=sf.meta[name]
        if tensor['shape']!=[VOCAB,D]: raise ValueError('Unexpected head shape')
        if tensor['dtype']!='BF16': raise RuntimeError(f'Expected original BF16 checkpoint head; got {tensor["dtype"]}')
        start,end=tensor['data_offsets']; beginning=sf.data_begin
        out.write(sf.mm[beginning+start:beginning+end])
        meta['head']['source_tensor']=name
        meta['head']['head_offset']=D*4
        meta['head']['row_bytes']=D*2
    meta['head']['bytes']=head.stat().st_size
    meta['head']['sha256']=sha256(head)
    if meta['head']['bytes']!=D*4+VOCAB*D*2:raise RuntimeError('Head size check failed')
    try:
        from transformers import AutoTokenizer
        tok=AutoTokenizer.from_pretrained(REPO, local_files_only=True)
        ids=tok.apply_chat_template([{'role':'user','content':args.prompt}],tokenize=True,add_generation_prompt=True)
        meta['prompt']={'text':args.prompt,'token_ids':ids,'token_count':len(ids),'chat_template':True}
        print('CHAT PROMPT TOKENS',len(ids),ids,flush=True)
    except ImportError:
        print('Tokenizer not installed: prompt tokens omitted; install transformers or supply from Phase 79',flush=True)
    manifest_path.write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print('PHASE80 EXPORT PASS',flush=True)
    print('TRANSFORMER',image, meta['transformer']['bytes'],meta['transformer']['sha256'],flush=True)
    print('HEAD',head,meta['head']['bytes'],meta['head']['sha256'],flush=True)
    print('NEXT: run --verify-only before any ESP32 integration',flush=True)

if __name__=='__main__':main()
