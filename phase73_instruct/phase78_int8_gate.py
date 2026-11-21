"""Phase 78: HF FP32 vs symmetric per-output-row W8A32, PC diagnostic only.

Weight-only INT8 across all transformer linear projections, FP32 embeddings,
normalization, activations, and LM head. Exact causal model structure via
Hugging Face; not a simulation of a future ESP32 integer kernel.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

REPO = 'Qwen/Qwen2.5-0.5B-Instruct'
PROJS = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')


def quantize_int8_per_row_numpy(w):
    """Symmetric, signed INT8, one absmax scale per output row; return dequantized FP32."""
    w = np.asarray(w, dtype=np.float32)
    if w.ndim != 2:
        raise ValueError('Weight must be a matrix')
    maxabs = np.max(np.abs(w), axis=1, keepdims=True)
    scale = np.where(maxabs == 0, 1.0, maxabs / 127.0).astype(np.float32)
    q = np.clip(np.rint(w / scale), -127, 127).astype(np.int8)
    return q.astype(np.float32) * scale


def quantize_int8_per_row_inplace(tensor):
    """Avoid storing an entire extra FP32 model or second set of weights."""
    import torch
    if tensor.ndim != 2 or tensor.dtype != torch.float32:
        raise ValueError('Requires FP32 matrix')
    scale = tensor.detach().abs().amax(dim=1, keepdim=True).div_(127)
    scale.clamp_(min=torch.finfo(torch.float32).tiny)
    q = torch.clamp(torch.round(tensor / scale), -127, 127).to(torch.int8)
    reconstructed = q.to(dtype=torch.float32).mul_(scale)
    abs_err = (tensor - reconstructed).abs()
    mae = float(abs_err.mean())
    maxerr = float(abs_err.max())
    tensor.copy_(reconstructed)
    return mae, maxerr


def stats(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    err = np.abs(a - b)
    tol = .001 + .001 * np.abs(a)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    cosine = float(np.dot(a.ravel(), b.ravel()) / (na * nb)) if na and nb else None
    return {'failures_atol001_rtol001': int((err > tol).sum()),
            'size': int(err.size), 'max_abs': float(err.max()),
            'mean_abs': float(err.mean()), 'cosine': cosine}


def evaluate(model, inputs, capture_hidden=True):
    import torch
    with torch.inference_mode():
        out = model(input_ids=inputs, use_cache=False, output_hidden_states=capture_hidden,
                    return_dict=True)
        logits = out.logits[0, -1].float()
        top = torch.topk(logits, k=2)
        token = int(top.indices[0])
        score = float(top.values[0])
        margin = float(top.values[0] - top.values[1])
        hidden = [h[0, -1, :].detach().cpu().numpy().astype(np.float32, copy=True)
                  for h in out.hidden_states] if capture_hidden else []
        return {'token': token, 'score': score, 'margin': margin}, hidden


def run(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if args.offline:
        os.environ['HF_HUB_OFFLINE'] = '1'
    print('PHASE78 | FULL INT8 WEIGHT-ONLY FIDELITY GATE | PC ONLY', flush=True)
    tok = AutoTokenizer.from_pretrained(REPO, local_files_only=args.offline)
    if args.chat:
        messages = [{'role':'user','content':args.prompt}]
        ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    else:
        ids = tok(args.prompt, add_special_tokens=False).input_ids
    if not ids:
        raise ValueError('Empty prompt')
    print('PROMPT:', repr(args.prompt), 'CHAT_TEMPLATE:', args.chat, 'TOKENS:',len(ids),flush=True)
    print('INPUT TOKEN IDS', ids, flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        REPO, torch_dtype=torch.float32, attn_implementation='eager',
        local_files_only=args.offline).eval().cpu()
    inputs = torch.tensor([ids], dtype=torch.long)
    print('Evaluating original FP32 checkpoint...', flush=True)
    original, orig_hidden = evaluate(model, inputs)
    print('ORIGINAL FP32 FIRST TOKEN', original['token'], repr(tok.decode([original['token']])),flush=True)
    print('Converting all 168 transformer projections to per-row INT8; head and embeddings remain FP32...', flush=True)
    matrix_errors = []
    with torch.no_grad():
        for index, layer in enumerate(model.model.layers, start=1):
            for name in PROJS:
                mod = getattr(layer.self_attn if name in PROJS[:4] else layer.mlp, name)
                mae, maxerr = quantize_int8_per_row_inplace(mod.weight)
                matrix_errors.append({'layer':index,'name':name,'weight_mae':mae,'weight_max_abs':maxerr})
            print(f'  quantized {index}/24 layers',flush=True)
    print('Evaluating full transformer with INT8-reconstructed projection weights...',flush=True)
    int8, int8_hidden = evaluate(model, inputs)
    print('INT8 FIRST TOKEN',int8['token'],repr(tok.decode([int8['token']])),flush=True)
    print('INT8 TOKEN MATCH',int8['token']==original['token'],flush=True)
    print('ORIGINAL MARGIN',original['margin'],'INT8 MARGIN',int8['margin'],flush=True)
    comparisons = []
    for layer_idx in range(1, len(orig_hidden)):
        measurement = stats(orig_hidden[layer_idx],int8_hidden[layer_idx])
        measurement['layer'] = layer_idx
        comparisons.append(measurement)
        print(f"  layer={layer_idx:2d} fails={measurement['failures_atol001_rtol001']:3d}/{measurement['size']} "
              f"max_abs={measurement['max_abs']:.8g} mean_abs={measurement['mean_abs']:.8g} "
              f"cosine={measurement['cosine']:.9f}",flush=True)
    report = {'checkpoint':REPO,'prompt':args.prompt,'chat_template':args.chat,'input_ids':ids,
              'original': original,'int8':int8,'token_match':int8['token']==original['token'],
              'quantization':'per-output-row symmetric INT8, W8A32; projections only',
              'head':'original FP32 LM head (not INT8)','layer_comparisons':comparisons,
              'projection_weight_errors':matrix_errors,
              'limitations':['PC simulation only: ESP32 INT8 kernels not implemented',
               'One-step greedy prediction; matching alone does not establish conversational quality',
               'Non-INT8 embedding/head/norm are still FP32',
               'No INT8 export format generated in this phase']}
    Path(args.report).write_text(json.dumps(report,indent=2),encoding='utf8')
    print('PHASE78 REPORT',args.report,flush=True)
    print('PHASE78 FIDELITY GATE', 'PASS' if report['token_match'] else 'FAIL',flush=True)
    return 0 if report['token_match'] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prompt',default='Hello')
    parser.add_argument('--chat',action='store_true',help='Use official Instruct user chat template (more tokens)')
    parser.add_argument('--offline',action='store_true',help='Force model loading from Hugging Face cache')
    parser.add_argument('--report',default='phase78_int8_fidelity_report.json')
    args = parser.parse_args()
    raise SystemExit(run(args))


if __name__ == '__main__':
    main()
