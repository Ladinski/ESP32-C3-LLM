"""Offline-only per-layer fidelity diagnostic: original Qwen Instruct vs exported INT4.
No firmware, serial connection, or ESP32 work. Does not overwrite any model image.
"""
import argparse
import hashlib
import json
import mmap
from pathlib import Path

REPO = 'Qwen/Qwen2.5-0.5B-Instruct'


def check_image(image, metadata, label):
    path=Path(image)
    meta=json.loads(Path(metadata).read_text(encoding='utf-8'))
    if meta.get('model')!=REPO:
        raise ValueError(f'{label} model metadata identifies {meta.get("model")}, not {REPO}')
    if path.stat().st_size!=meta['image_size']:
        raise ValueError(f'{label} image size mismatch')
    with path.open('rb') as stream:
        digest=hashlib.file_digest(stream,'sha256').hexdigest()
    if digest!=meta['sha256']:
        raise ValueError(f'{label} SHA256 mismatch')
    print(f'{label} VERIFIED {digest}',flush=True)
    return meta


def measures(a,b,atol,rtol):
    import numpy as np
    x=np.asarray(a,dtype=np.float64).reshape(-1)
    y=np.asarray(b,dtype=np.float64).reshape(-1)
    if x.shape!=y.shape or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError(f'Invalid comparison shapes or values: {x.shape}, {y.shape}')
    diff=np.abs(x-y)
    denom=np.linalg.norm(x)*np.linalg.norm(y)
    cosine=float(np.dot(x,y)/denom) if denom else float('nan')
    return dict(failures=int(np.count_nonzero(diff > atol+rtol*np.abs(x))),count=int(x.size),
                max_abs=float(diff.max()),mean_abs=float(diff.mean()),
                max_normalized=float(np.max(diff/np.maximum(np.abs(x),1))),cosine=cosine)


def compare(args):
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from phase69_reference_math import reference
    from export_phase45_head import BFloatReader
    from huggingface_hub import hf_hub_download

    if not 1<=args.layers<=24: raise ValueError('--layers must be between 1 and 24')
    model_meta=check_image(args.image,args.metadata,'TRANSFORMER')
    head_meta=check_image(args.head,args.head_metadata,'OUTPUT HEAD')
    if len(model_meta['layers'])<args.layers: raise ValueError('Not enough INT4 layers')
    checkpoint=Path(args.checkpoint or hf_hub_download(REPO,'model.safetensors'))
    tokenizer=AutoTokenizer.from_pretrained(REPO)
    ids=tokenizer(args.prompt,add_special_tokens=False).input_ids
    if len(ids)!=1: raise ValueError('This diagnostic requires one plain-text input token (try --prompt Hello)')
    token_id=ids[0]
    print('PHASE74 | ORIGINAL vs INT4 LAYER AUDIT | PC ONLY',flush=True)
    print('PROMPT',repr(args.prompt),'INPUT TOKEN ID',token_id,flush=True)
    with BFloatReader(checkpoint) as source:
        embedding=np.asarray(source.get_row('model.embed_tokens.weight',token_id),dtype=np.float32).copy()
    print('Loading official Instruct model for single-token layer audit...',flush=True)
    model=AutoModelForCausalLM.from_pretrained(REPO,torch_dtype=torch.float32,attn_implementation='eager')
    model.eval().to('cpu')
    if len(model.model.layers)!=24: raise ValueError('Original model does not have 24 layers')
    original={}
    hooks=[]
    for i,layer in enumerate(model.model.layers):
        def take_output(module, inputs, output, index=i):
            value=output[0] if isinstance(output,tuple) else output
            original[index]=value[0,-1].detach().cpu().numpy().astype(np.float32).copy()
        hooks.append(layer.register_forward_hook(take_output))
    with torch.inference_mode():
        original_logits=model(input_ids=torch.tensor([[token_id]]),use_cache=False).logits[0,-1].detach().cpu().numpy().copy()
    for h in hooks: h.remove()
    original_id=int(np.argmax(original_logits))
    print('ORIGINAL MODEL FIRST TOKEN',original_id,repr(tokenizer.decode([original_id])),flush=True)
    del model
    with open(args.image,'rb') as file:
        mm=mmap.mmap(file.fileno(),0,access=mmap.ACCESS_READ)
        try:
            print('Computing independent INT4 replay (24 layers)...',flush=True)
            replay=reference(mm,model_meta['layers'],[embedding])
        finally:
            mm.close()
    report=[]
    first=None
    print('PHASE74 LAYER-BY-LAYER COMPARISON: atol',args.atol,'rtol',args.rtol,flush=True)
    for i in range(args.layers):
        r=measures(original[i],replay[(0,i)],args.atol,args.rtol)
        r['layer']=i+1
        report.append(r)
        if first is None and r['failures']>0: first=i+1
        print(f'  layer={i+1:2d} failures={r["failures"]:3d}/{r["count"]} '
              f'max_abs={r["max_abs"]:.9g} mean_abs={r["mean_abs"]:.9g} '
              f'max_norm={r["max_normalized"]:.9g} cosine={r["cosine"]:.9f}',flush=True)
    print('PHASE74 FIRST FAILING LAYER',first if first else 'NONE',flush=True)
    print('PHASE74 ORIGINAL FIRST TOKEN',original_id,repr(tokenizer.decode([original_id])),flush=True)
    print('PHASE74 NOTE: compares original float32 transformer against per-row INT4; cannot alone distinguish quantization loss from mathematical implementation error.',flush=True)
    payload={'prompt':args.prompt,'token_id':token_id,'original_first_token':original_id,
             'first_failing_layer':first,'layers':report,'atol':args.atol,'rtol':args.rtol,
             'transformer_sha256':model_meta['sha256'],'head_sha256':head_meta['sha256']}
    Path(args.report).write_text(json.dumps(payload,indent=2),encoding='utf-8')
    print('PHASE74 REPORT SAVED',args.report,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prompt',default='Hello')
    p.add_argument('--layers',type=int,default=24)
    p.add_argument('--atol',type=float,default=.001)
    p.add_argument('--rtol',type=float,default=.001)
    p.add_argument('--image',default='phase25_qwen_24layers_int4.bin')
    p.add_argument('--metadata',default='phase25_qwen_24layers_int4.json')
    p.add_argument('--head',default='phase45_qwen_head_int4.bin')
    p.add_argument('--head-metadata',default='phase45_qwen_head_int4.json')
    p.add_argument('--checkpoint')
    p.add_argument('--report',default='phase74_layer_fidelity_report.json')
    args=p.parse_args()
    compare(args)

if __name__=='__main__':main()
