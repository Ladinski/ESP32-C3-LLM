"""Phase 75: FP32 original-vs-INT4 exported projection attribution on identical activations.
PC diagnostic only. Does not use or change ESP32 firmware, model files, or serial port.
This tests individual projection fidelity, NOT an end-to-end corrected INT4 model.
"""
import argparse
import hashlib
import json
import mmap
from pathlib import Path

REPO = 'Qwen/Qwen2.5-0.5B-Instruct'
MATRIX_NAMES = ('q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj')


def unpack_rows(block, nrows, cols):
    """Decode consecutive 4-bit signed packed matrix rows and row float32 scales."""
    import numpy as np
    width = cols // 2 + 4
    if cols % 2 or len(block) != nrows*width:
        raise ValueError('Wrong packed row size')
    b = np.frombuffer(block,dtype=np.uint8).reshape(nrows,width)
    packed = b[:,:cols//2]
    low = (packed & 15).astype(np.int8)
    high = (packed >> 4).astype(np.int8)
    low[low>=8] -= 16
    high[high>=8] -= 16
    w = np.empty((nrows,cols),dtype=np.float32)
    w[:,0::2] = low
    w[:,1::2] = high
    scales = np.frombuffer(b[:,cols//2:].copy().tobytes(), dtype='<f4')
    w *= scales[:,None]
    return w


def int4_matvec(mm,mat,x,rows_per_block=16,original_weight=None):
    import numpy as np
    rows=int(mat['rows']);cols=int(mat['cols']);width=int(mat['row_bytes'])
    x=np.asarray(x,dtype=np.float32).reshape(-1)
    if x.size!=cols: raise ValueError('Input dimension mismatch')
    out=np.empty(rows,dtype=np.float32)
    weight_mae_sum=0.;weight_count=0;weight_max=0.
    for start in range(0,rows,rows_per_block):
        end=min(rows,start+rows_per_block)
        raw=mm[mat['offset']+start*width:mat['offset']+end*width]
        w=unpack_rows(raw,end-start,cols)
        out[start:end]=w@x
        if original_weight is not None:
            orig=original_weight[start:end]
            err=np.abs(w.astype(np.float64)-orig.astype(np.float64))
            weight_mae_sum+=float(err.sum());weight_count+=err.size
            weight_max=max(weight_max,float(err.max()))
    return out,dict(mae=weight_mae_sum/weight_count,max_abs=weight_max) if weight_count else None


def vector_metrics(ref,other,atol=.001,rtol=.001):
    import numpy as np
    a=np.asarray(ref,dtype=np.float64).reshape(-1)
    b=np.asarray(other,dtype=np.float64).reshape(-1)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Invalid vector comparison')
    d=np.abs(a-b)
    norm=np.linalg.norm(a)*np.linalg.norm(b)
    return dict(count=int(a.size),failures=int(np.count_nonzero(d>atol+rtol*np.abs(a))),
                max_abs=float(d.max()),mean_abs=float(d.mean()),
                cosine=float(np.dot(a,b)/norm) if norm else None)


def verify_image(args):
    meta=json.loads(Path(args.metadata).read_text(encoding='utf8'))
    if meta.get('model') != REPO: raise ValueError('Not the Instruct export')
    if len(meta['layers'])!=24: raise ValueError('Not a 24-layer image')
    if Path(args.image).stat().st_size != meta['image_size']: raise ValueError('Wrong image size')
    with open(args.image,'rb') as f: actual=hashlib.file_digest(f,'sha256').hexdigest()
    if actual!=meta['sha256']: raise ValueError('Image SHA256 mismatch')
    return meta,actual


def run(args):
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import hf_hub_download
    from export_qwen_layer0 import BFloatReader

    selected=sorted(set(int(part) for part in args.layers.split(',')))
    if not selected or min(selected)<1 or max(selected)>24: raise ValueError('Select layers 1..24')
    meta,sha=verify_image(args)
    print('PHASE75 | INSTRUCT FP32 vs INT4 SAME-INPUT PROJECTION AUDIT | PC ONLY',flush=True)
    print('INT4 IMAGE VERIFIED',sha,flush=True)
    print('LAYERS',selected,'NO ESP32 / COM3 / FIRMWARE CHANGES',flush=True)
    tokenizer=AutoTokenizer.from_pretrained(REPO)
    ids=tokenizer(args.prompt,add_special_tokens=False).input_ids
    if len(ids)!=1:raise ValueError('Use a single tokenizer input token, e.g. --prompt Hello')
    print('INPUT TOKEN',ids[0],repr(args.prompt),flush=True)
    model=AutoModelForCausalLM.from_pretrained(REPO,torch_dtype=torch.float32,attn_implementation='eager')
    model.eval().to('cpu')
    original={}
    hooks=[]
    for layernum in selected:
        layer=model.model.layers[layernum-1]
        for name in MATRIX_NAMES:
            group=layer.self_attn if name in MATRIX_NAMES[:4] else layer.mlp
            module=getattr(group,name)
            def pre(mod, input_tuple, layernum=layernum, name=name):
                original[layernum,name,'input']=input_tuple[0][0,-1].detach().cpu().numpy().astype(np.float32).copy()
            def post(mod, input_tuple, output, layernum=layernum, name=name):
                original[layernum,name,'output']=output[0,-1].detach().cpu().numpy().astype(np.float32).copy()
            hooks.append(module.register_forward_pre_hook(pre))
            hooks.append(module.register_forward_hook(post))
    with torch.inference_mode():
        logits=model(input_ids=torch.tensor([ids]),use_cache=False).logits[0,-1]
        print('ORIGINAL GREEDY TOKEN',int(logits.argmax()),repr(tokenizer.decode([int(logits.argmax())])),flush=True)
    for h in hooks:h.remove()

    results=[]
    first_severe=None
    with open(args.image,'rb') as fp, BFloatReader(args.checkpoint or hf_hub_download(REPO,'model.safetensors')) as source:
        mm=mmap.mmap(fp.fileno(),0,access=mmap.ACCESS_READ)
        try:
            for ln in selected:
                layer_meta=meta['layers'][ln-1]
                matrices={m['name']:m for m in layer_meta['matrices']}
                print(f'\nLAYER {ln}: original projection output vs INT4 same-input output',flush=True)
                for name in MATRIX_NAMES:
                    x=original[ln,name,'input'];y=original[ln,name,'output']
                    group='self_attn' if name in MATRIX_NAMES[:4] else 'mlp'
                    key=f'model.layers.{ln-1}.{group}.{name}.weight'
                    orig_weight=source.get_tensor(key) if args.weight_error else None
                    predicted,wm=int4_matvec(mm,matrices[name],x,original_weight=orig_weight)
                    if group=='self_attn' and name in ('q_proj','k_proj','v_proj'):
                        bias=source.get_tensor(f'model.layers.{ln-1}.self_attn.{name}.bias')
                        predicted+=bias.astype(np.float32)
                    metrics=vector_metrics(y,predicted,args.atol,args.rtol)
                    metrics.update(layer=ln,projection=name,weight_error=wm)
                    results.append(metrics)
                    print(f'  {name:<10} fails={metrics["failures"]:4}/{metrics["count"]:<4} '
                          f'max={metrics["max_abs"]:.7g} mean={metrics["mean_abs"]:.7g} '
                          f'cosine={metrics["cosine"]:.7f}'+
                          (f' weight_mae={wm["mae"]:.7g}' if wm else ''),flush=True)
                    if first_severe is None and metrics['failures']>0:
                        first_severe={'layer':ln,'projection':name}
        finally:mm.close()
    print('\nFIRST SAME-INPUT QUANTIZATION TOLERANCE FAILURE',first_severe,flush=True)
    print('INTERPRETATION: these errors compare original FP32 and exported INT4 weights with IDENTICAL original inputs.',flush=True)
    print('They do not include cumulative activation drift or prove a bug in the ESP32 implementation.',flush=True)
    print('PHASE75 DIAGNOSTIC COMPLETE',flush=True)
    Path(args.report).write_text(json.dumps(dict(model=REPO,prompt=args.prompt,token_id=ids[0],
        layers=selected,int4_sha256=sha,first_failure=first_severe,results=results,
        note='Same original input activations for every projection; not a sequential INT4 replay.'),indent=2),encoding='utf8')
    print('REPORT',args.report,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prompt',default='Hello')
    p.add_argument('--layers',default='1,3,22,23,24',help='Comma-separated 1-based transformer layer numbers')
    p.add_argument('--image',default='phase25_qwen_24layers_int4.bin')
    p.add_argument('--metadata',default='phase25_qwen_24layers_int4.json')
    p.add_argument('--checkpoint',default=None)
    p.add_argument('--weight-error',action='store_true',help='Also measure original vs reconstructed weight deviations (slower)')
    p.add_argument('--atol',type=float,default=.001)
    p.add_argument('--rtol',type=float,default=.001)
    p.add_argument('--report',default='phase75_projection_attribution_report.json')
    args=p.parse_args()
    run(args)

if __name__=='__main__':main()
