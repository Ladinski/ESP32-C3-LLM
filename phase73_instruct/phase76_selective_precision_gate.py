"""Phase 76: PC-only end-to-end selective precision experiments for Qwen2.5-0.5B-Instruct.

The original HF model provides architecture, norm, embedding and attention semantics.
Every transformer Linear weight is replaced at inference by EXACT exported per-row INT4
reconstruction, except selected DOWN projections re-quantized to INT8 or FP16 from
pretrained checkpoint. The output head is the EXISTING exported INT4 head via the
Phase 69 reference. No ESP32 computations or firmware alterations.

IMPORTANT: An INT4 baseline that fails to reproduce the Phase73 INT4 result (52643)
invalidates conclusions about device-model improvements, and is flagged.
"""
import argparse
import hashlib
import json
import mmap
import os
from pathlib import Path

import numpy as np

REPO = 'Qwen/Qwen2.5-0.5B-Instruct'
PROJS=('q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj')
SCENARIOS={
 'baseline': {},
 'l3_fp16': {(3,'down_proj'):'fp16'},
 'l22_fp16': {(22,'down_proj'):'fp16'},
 'both_fp16': {(3,'down_proj'):'fp16',(22,'down_proj'):'fp16'},
 'both_int8': {(3,'down_proj'):'int8',(22,'down_proj'):'int8'},
 'late_fp16': {(3,'down_proj'):'fp16',**{(k,'down_proj'):'fp16' for k in (22,23,24)}}
}


def validate_file(path,meta):
    if Path(path).stat().st_size!=meta['image_size']:
        raise ValueError(f'Wrong byte count for {path}')
    with open(path,'rb') as f: actual=hashlib.file_digest(f,'sha256').hexdigest()
    if actual!=meta['sha256']:
        raise ValueError(f'Unexpected SHA256 for {path}: {actual}')
    if meta.get('model')!=REPO:
        raise ValueError(f'Image {path} does not belong to {REPO}')
    return actual


def unpack_int4(mm,meta):
    rows,cols=int(meta['rows']),int(meta['cols'])
    width=int(meta['row_bytes'])
    if cols%2 or width!=cols//2+4:raise ValueError('Unknown INT4 row layout')
    raw=np.frombuffer(mm[meta['offset']:meta['offset']+rows*width],dtype=np.uint8).reshape(rows,width)
    p=raw[:,:cols//2]
    w=np.empty((rows,cols),dtype=np.float32)
    w[:,::2]=(p&15).astype(np.float32)
    w[:,1::2]=(p>>4).astype(np.float32)
    w[w>=8]-=16
    scale=np.frombuffer(raw[:,cols//2:].copy().tobytes(),dtype='<f4')
    w*=scale[:,None]
    return w


def quantize_int8(weight):
    weight=np.asarray(weight,dtype=np.float32)
    scales=np.max(np.abs(weight),axis=1,keepdims=True)/127.0
    scales=np.where(scales==0,1.0,scales).astype(np.float32)
    q=np.clip(np.rint(weight/scales),-127,127).astype(np.int8)
    return q.astype(np.float32)*scales


def build_run(model,mm,metadata,override):
    """Patch all 168 Linear projections with exact INT4, selected original high precision."""
    import torch
    from torch.nn import functional as F
    handles=[]
    for ln, layer_meta in enumerate(metadata['layers'],start=1):
        matrices={m['name']:m for m in layer_meta['matrices']}
        layer=model.model.layers[ln-1]
        for name in PROJS:
            module=getattr(layer.self_attn if name in PROJS[:4] else layer.mlp,name)
            precision=override.get((ln,name),'int4')
            if precision=='int4':
                # Use the signed-4-bit weight bytes from the exact exported image.
                w=unpack_int4(mm,matrices[name])
            elif precision=='fp16':
                w=module.weight.detach().cpu().numpy().astype(np.float16).astype(np.float32)
            elif precision=='int8':
                w=quantize_int8(module.weight.detach().cpu().numpy())
            else:
                raise ValueError(precision)
            w_tensor=torch.from_numpy(w)
            # A closure keeps each tensor alive through its registered hook.
            def patched(mod,inputs,original_output,W=w_tensor):
                return F.linear(inputs[0],W,mod.bias)
            handles.append(module.register_forward_hook(patched))
    return handles


def run(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from phase69_full_reference import head_reference

    # Offline loading avoids HF HTTP HEAD retries; Phase72 cached both models.
    os.environ['HF_HUB_OFFLINE']='1'
    scenarios=[x.strip() for x in args.scenarios.split(',') if x.strip()]
    if not scenarios or any(x not in SCENARIOS for x in scenarios):
        raise ValueError(f'Valid scenarios: {",".join(SCENARIOS)}')
    model_meta=json.loads(Path(args.metadata).read_text(encoding='utf8'))
    head_meta=json.loads(Path(args.head_metadata).read_text(encoding='utf8'))
    if len(model_meta['layers'])!=24 or head_meta.get('vocab')!=151936:
        raise ValueError('Expected complete 24-layer Qwen model')
    msha=validate_file(args.image,model_meta)
    hsha=validate_file(args.head,head_meta)
    print('PHASE76 SELECTIVE-PRECISION FIDELITY GATE | PC ONLY',flush=True)
    print('TRANSFORMER SHA256',msha,'HEAD SHA256',hsha,flush=True)
    print('Loading original architecture and parameters from local HF cache...',flush=True)
    tokenizer=AutoTokenizer.from_pretrained(REPO,local_files_only=True)
    ids=tokenizer(args.prompt,add_special_tokens=False).input_ids
    if len(ids)!=1:raise ValueError(f'One raw token only; got {ids}. Use --prompt Hello')
    model=AutoModelForCausalLM.from_pretrained(REPO,torch_dtype=torch.float32,attn_implementation='eager',local_files_only=True)
    model.eval().cpu()
    inp=torch.tensor([ids],dtype=torch.long)
    with torch.inference_mode():
        baseline_logits=model(input_ids=inp,use_cache=False).logits[0,-1]
        orig_id=int(baseline_logits.argmax())
    print('ORIGINAL FP32 HEAD TOKEN',orig_id,repr(tokenizer.decode([orig_id])),flush=True)
    result=[]
    with open(args.image,'rb') as f,open(args.head,'rb') as hf:
        mm=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ)
        hm=mmap.mmap(hf.fileno(),0,access=mmap.ACCESS_READ)
        try:
            for i,scenario in enumerate(scenarios,1):
                print(f'\nSCENARIO {i}/{len(scenarios)}: {scenario}',flush=True)
                handles=build_run(model,mm,model_meta,SCENARIOS[scenario])
                hidden=[]
                def capture_input(mod, inputs):
                    hidden.append(inputs[0][0,-1].detach().cpu().numpy().astype(np.float32).copy())
                handles.append(model.model.norm.register_forward_pre_hook(capture_input))
                try:
                    with torch.inference_mode():
                        logits=model(input_ids=inp,use_cache=False).logits[0,-1]
                        fp32_head_id=int(logits.argmax())
                        if len(hidden)!=1:raise RuntimeError(f'Unexpected final norm input count {len(hidden)}')
                        qid,score,second=head_reference(hm,hidden[0],head_meta)
                finally:
                    for h in handles:h.remove()
                passed=(qid==orig_id)
                print('  Original FP32 token:',orig_id,repr(tokenizer.decode([orig_id])),flush=True)
                print('  Hybrid + original FP32 head:',fp32_head_id,repr(tokenizer.decode([fp32_head_id])),flush=True)
                print('  Hybrid + exported INT4 head:',qid,repr(tokenizer.decode([qid])),flush=True)
                print('  Exported INT4 head margin:',round(float(score-second),5),'AGREES WITH ORIGINAL:',passed,flush=True)
                result.append(dict(scenario=scenario,fp32_head_token=fp32_head_id,int4_head_token=qid,
                                   int4_head_margin=float(score-second),matches_original=passed))
                if scenario=='baseline':
                    print('  BASELINE CROSS-CHECK EXPECTED ID 52643:', 'PASS' if qid==52643 else 'MISMATCH',flush=True)
        finally:
            hm.close();mm.close()
    base=next((x for x in result if x['scenario']=='baseline'),None)
    report={'checkpoint':REPO,'prompt':args.prompt,'input_ids':ids,'original_token':orig_id,
            'previous_phase73_baseline_token':52643,'baseline_crosscheck':None if base is None else base['int4_head_token']==52643,
            'scenarios':result,'limitations':[
            'One raw prompt token without Instruct chat template',
            'PC-only mixed-precision simulation; ESP32 has no added INT8/FP16 kernels',
            'Quantized transformer uses model-provided attention, norms, and embeddings',
            'Must reproduce prior baseline before claiming improvements']}
    Path(args.report).write_text(json.dumps(report,indent=2),encoding='utf8')
    print('\nPHASE76 REPORT',args.report,flush=True)
    if base is not None and base['int4_head_token']!=52643:
        print('PHASE76 BASELINE MISMATCH: DO NOT interpret selective precision cases as validated improvements.',flush=True)
        return 2
    winners=[r['scenario'] for r in result if r['matches_original'] and r['scenario']!='baseline']
    print('PHASE76 MATCHING SCENARIOS',winners if winners else 'NONE',flush=True)
    print('PHASE76 COMPLETE. No ESP32 firmware or weights changed.',flush=True)
    return 0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prompt',default='Hello')
    p.add_argument('--scenarios',default=','.join(SCENARIOS))
    p.add_argument('--image',default='phase25_qwen_24layers_int4.bin')
    p.add_argument('--metadata',default='phase25_qwen_24layers_int4.json')
    p.add_argument('--head',default='phase45_qwen_head_int4.bin')
    p.add_argument('--head-metadata',default='phase45_qwen_head_int4.json')
    p.add_argument('--report',default='phase76_selective_precision_report.json')
    raise SystemExit(run(p.parse_args()))

if __name__=='__main__':main()
