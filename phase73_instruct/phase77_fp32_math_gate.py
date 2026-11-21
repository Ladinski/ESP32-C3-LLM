"""Phase 77: independent NumPy FP32 Qwen2.5-Instruct one-token audit.

Reads ORIGINAL checkpoint tensors from local safetensors and implements the
single-token transformer in NumPy, with no INT4 weights and NO HF model.forward
calls within the NumPy path. Compares each layer to Hugging Face's implementation.
One token only: causal softmax is exactly 1, so Q/K/RoPE cannot be checked here.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

REPO = 'Qwen/Qwen2.5-0.5B-Instruct'
PROJS = ('q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj')


def rms(x, gamma, eps):
    # Intentionally matches Qwen2RMSNorm PyTorch float32 semantics.
    variance = np.mean(np.square(x, dtype=np.float32), dtype=np.float32)
    return np.asarray(x * np.float32(1.0 / np.sqrt(np.float32(variance + eps))),dtype=np.float32) * gamma


def silu(x):
    x = np.asarray(x,dtype=np.float32)
    return np.asarray(x / (np.float32(1.0) + np.exp(-np.clip(x,-80,80))),dtype=np.float32)


def quality(ref, actual, atol, rtol):
    ref = np.asarray(ref,dtype=np.float32).reshape(-1)
    actual = np.asarray(actual,dtype=np.float32).reshape(-1)
    if ref.shape != actual.shape or not np.isfinite(actual).all():
        raise ValueError('Invalid tensor shape or nonfinite output')
    delta = np.abs(ref.astype(np.float64)-actual.astype(np.float64))
    dot=float(np.dot(ref.astype(np.float64),actual.astype(np.float64)))
    norm=float(np.linalg.norm(ref.astype(np.float64))*np.linalg.norm(actual.astype(np.float64)))
    return {'fails':int(np.count_nonzero(delta>atol+rtol*np.abs(ref))),
            'count':int(len(ref)),'max_abs':float(delta.max()),
            'mean_abs':float(delta.mean()),
            'cosine': float(dot/norm) if norm else None}


def original_states(model, token_id):
    import torch
    states={}
    taps=[]
    for i,layer in enumerate(model.model.layers):
        def record(module, args, output, layer_no=i):
            val=output[0] if isinstance(output,tuple) else output
            states[layer_no]=val[0,-1].detach().cpu().numpy().copy()
        taps.append(layer.register_forward_hook(record))
    with torch.inference_mode():
        logits=model(input_ids=torch.tensor([[token_id]],dtype=torch.long),use_cache=False).logits[0,-1].detach().cpu()
    for tap in taps:tap.remove()
    return states, logits


def independent_one_token(source, embedding, layers, reference, eps, atol, rtol):
    # Entirely separate from HF forward; only original safetensors are consulted.
    state=np.asarray(embedding,dtype=np.float32).copy()
    stats=[]
    for layer_i in range(layers):
        pre=f'model.layers.{layer_i}.'
        n1=source.get_tensor(pre+'input_layernorm.weight').astype(np.float32)
        n2=source.get_tensor(pre+'post_attention_layernorm.weight').astype(np.float32)
        attn=rms(state,n1,eps)
        # Exactly one input token: attention probability for this sole key = 1.
        # Q/K/RoPE don't affect output, but V and O do.
        v=np.asarray(source.get_tensor(pre+'self_attn.v_proj.weight') @ attn,dtype=np.float32)
        v += source.get_tensor(pre+'self_attn.v_proj.bias').astype(np.float32)
        # Repeat each KV head for 7 query heads: 2 x 7 x 64 = 896.
        context=np.repeat(v.reshape(2,64),7,axis=0).reshape(896)
        o=np.asarray(source.get_tensor(pre+'self_attn.o_proj.weight') @ context,dtype=np.float32)
        residual=np.asarray(state+o,dtype=np.float32)
        mlp_input=rms(residual,n2,eps)
        gate=np.asarray(source.get_tensor(pre+'mlp.gate_proj.weight') @ mlp_input,dtype=np.float32)
        up=np.asarray(source.get_tensor(pre+'mlp.up_proj.weight') @ mlp_input,dtype=np.float32)
        activation=np.asarray(silu(gate)*up,dtype=np.float32)
        down=np.asarray(source.get_tensor(pre+'mlp.down_proj.weight') @ activation,dtype=np.float32)
        state=np.asarray(residual+down,dtype=np.float32)
        m=quality(reference[layer_i],state,atol,rtol)
        stats.append({'layer':layer_i+1,**m})
        print(f'  layer={layer_i+1:2d} fails={m["fails"]:3d}/896 '
              f'max_abs={m["max_abs"]:.8g} mean_abs={m["mean_abs"]:.8g} '
              f'cosine={m["cosine"]:.9f}',flush=True)
    return state,stats


def run(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from export_phase45_head import BFloatReader
    from huggingface_hub import hf_hub_download
    if args.layers < 1 or args.layers > 24:raise ValueError('--layers must be 1..24')
    os.environ['HF_HUB_OFFLINE']='1'
    tok=AutoTokenizer.from_pretrained(REPO,local_files_only=True)
    ids=tok(args.prompt,add_special_tokens=False).input_ids
    if len(ids)!=1: raise ValueError(f'Exactly one token required for this mathematical gate, got {ids}')
    print('PHASE77 INDEPENDENT FP32 NUMPY VS HF | PC ONLY',flush=True)
    print('TOKEN',ids[0],repr(tok.decode(ids)),flush=True)
    checkpoint=Path(args.checkpoint) if args.checkpoint else Path(hf_hub_download(REPO,'model.safetensors',local_files_only=True))
    print('ORIGINAL CHECKPOINT',checkpoint,flush=True)
    model=AutoModelForCausalLM.from_pretrained(REPO,torch_dtype=torch.float32,attn_implementation='eager',local_files_only=True)
    model.eval().cpu()
    eps=float(model.config.rms_norm_eps)
    reference,orig_logits=original_states(model,ids[0])
    orig_token=int(torch.argmax(orig_logits).item())
    print('ORIGINAL TOKEN',orig_token,repr(tok.decode([orig_token])),flush=True)
    with BFloatReader(checkpoint) as source:
        embed=source.get_row('model.embed_tokens.weight',ids[0]).copy()
        print('NUMPY SINGLE-TOKEN FP32 REPLAY START',flush=True)
        state,stats=independent_one_token(source,embed,args.layers,reference,eps,args.atol,args.rtol)
        if args.layers==24:
            final_norm=source.get_tensor('model.norm.weight')
            normed=rms(state,final_norm,eps)
            # Output head ONLY is evaluated with model's original FP32 head;
            # all NumPy transformer computations are independent.
            with torch.inference_mode():
                numpy_logits=model.lm_head(torch.from_numpy(normed.copy()).reshape(1,1,-1))[0,-1]
            numpy_token=int(numpy_logits.argmax().item())
            print('NUMPY FP32 TRANSFORMER + ORIGINAL FP32 HEAD TOKEN',numpy_token,
                  repr(tok.decode([numpy_token])),flush=True)
            print('PHASE77 TOKEN MATCH',numpy_token==orig_token,flush=True)
        else:
            numpy_token=None
    first=next((x['layer'] for x in stats if x['fails']>0),None)
    print('PHASE77 FIRST FAILING LAYER',first if first is not None else 'NONE',flush=True)
    print('PHASE77 MATH GATE', 'PASS' if not first and (numpy_token is None or numpy_token==orig_token) else 'INVESTIGATE',flush=True)
    print('LIMIT: single-token attention cannot validate Q/K/RoPE, KV cache, or chat templating.',flush=True)
    report={'checkpoint':REPO,'prompt':args.prompt,'token_ids':ids,'layers':stats,
            'first_failing_layer':first,'original_token':orig_token,'numpy_fp32_token':numpy_token,
            'head_validation':'NumPy FP32 final hidden tested using original PyTorch FP32 lm_head',
            'limitations':['one raw input token only','Q/K and RoPE irrelevant to length-one attention',
                           'No ESP32 or quantized operations','NumPy vs Torch may have small FP32 rounding drift']}
    Path(args.report).write_text(json.dumps(report,indent=2),encoding='utf8')
    print('PHASE77 REPORT',args.report,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prompt',default='Hello')
    p.add_argument('--layers',type=int,default=24)
    p.add_argument('--atol',type=float,default=.001)
    p.add_argument('--rtol',type=float,default=.001)
    p.add_argument('--checkpoint',help='Path to the original Instruct model.safetensors')
    p.add_argument('--report',default='phase77_fp32_math_report.json')
    run(p.parse_args())

if __name__=='__main__':main()
