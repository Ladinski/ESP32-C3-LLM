"""Phase 79: compare greedy FP32 vs per-row W8A32 Qwen-Instruct chat generation.

PC diagnostic only. Reuses Phase 78's exact quantization method, tests autoregressive
output and retains all neural computation on PC for this validation only.
"""
import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from phase78_int8_gate import PROJS, REPO, quantize_int8_per_row_inplace


def generate(model, input_ids, max_new_tokens, eos_ids):
    """Greedy autoregressive generation, full-prefix recalculation for simplicity.

    Full-prefix recomputation avoids implementation-specific KV-cache behavior here.
    We are validating weight-only INT8 fidelity, not cache correctness or speed.
    """
    ids = input_ids.clone()
    tokens = []
    margins = []
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            output = model(input_ids=ids, use_cache=False, return_dict=True)
            top = torch.topk(output.logits[0, -1].float(), k=2)
            next_id = int(top.indices[0])
            margin = float(top.values[0] - top.values[1])
            tokens.append(next_id)
            margins.append(margin)
            ids = torch.cat((ids, torch.tensor([[next_id]], dtype=torch.long)), dim=1)
            if next_id in eos_ids:
                break
    return tokens, margins


def quantize_model(model):
    weight_errors = []
    with torch.no_grad():
        for layer_i, layer in enumerate(model.model.layers, 1):
            for name in PROJS:
                m = getattr(layer.self_attn if name in PROJS[:4] else layer.mlp, name)
                mae, mx = quantize_int8_per_row_inplace(m.weight)
                weight_errors.append({'layer': layer_i, 'projection': name,
                                      'mae': mae, 'max_abs': mx})
            print(f'  INT8 quantized {layer_i}/{len(model.model.layers)} layers', flush=True)
    return weight_errors


def compare(a, b):
    first_diff = next((i + 1 for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
    exact = a == b
    common = 0
    for x, y in zip(a, b):
        if x != y:
            break
        common += 1
    return {'exact_match': exact, 'matching_prefix_tokens': common,
            'first_different_generated_position': first_diff,
            'fp32_length': len(a), 'int8_length': len(b)}


def run(args):
    if args.max_new_tokens < 1 or args.max_new_tokens > 64:
        raise ValueError('--max-new-tokens must be between 1 and 64')
    if args.offline:
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
    print('PHASE79 | INT8 MULTITOKEN CHAT FIDELITY | PC ONLY', flush=True)
    print('MODEL', REPO, 'PROMPT', repr(args.prompt), 'MAX_NEW_TOKENS', args.max_new_tokens, flush=True)
    tokenizer = AutoTokenizer.from_pretrained(REPO, local_files_only=args.offline)
    ids = tokenizer.apply_chat_template([{'role': 'user', 'content': args.prompt}],
                                        tokenize=True, add_generation_prompt=True)
    if not isinstance(ids, list) or not ids:
        raise RuntimeError('Chat template returned no input IDs')
    print('CHAT INPUT TOKEN COUNT', len(ids), 'IDS', ids, flush=True)
    inputs = torch.tensor([ids], dtype=torch.long)
    model = AutoModelForCausalLM.from_pretrained(
        REPO, torch_dtype=torch.float32, attn_implementation='eager',
        local_files_only=args.offline).eval().cpu()
    eos_ids = {int(x) for x in ([model.generation_config.eos_token_id]
               if isinstance(model.generation_config.eos_token_id, int)
               else (model.generation_config.eos_token_id or []))}
    if tokenizer.eos_token_id is not None:
        eos_ids.add(int(tokenizer.eos_token_id))
    print('FP32 REFERENCE GENERATING...', flush=True)
    fp_tokens, fp_margins = generate(model, inputs, args.max_new_tokens, eos_ids)
    print('FP32 GENERATED IDS', fp_tokens, flush=True)
    print('FP32 GENERATED TEXT', repr(tokenizer.decode(fp_tokens, skip_special_tokens=True)), flush=True)
    print('APPLYING PER-ROW INT8 TO ALL 168 TRANSFORMER PROJECTIONS...', flush=True)
    matrix_errors = quantize_model(model)
    print('INT8 RECONSTRUCTED WEIGHTS GENERATING...', flush=True)
    int_tokens, int_margins = generate(model, inputs, args.max_new_tokens, eos_ids)
    print('INT8 GENERATED IDS', int_tokens, flush=True)
    print('INT8 GENERATED TEXT', repr(tokenizer.decode(int_tokens, skip_special_tokens=True)), flush=True)
    result = compare(fp_tokens, int_tokens)
    report = {'checkpoint': REPO, 'prompt': args.prompt, 'chat_template': True,
              'prompt_token_count': len(ids), 'prompt_ids': ids,
              'maximum_generated_tokens': args.max_new_tokens,
              'fp32': {'tokens': fp_tokens, 'text': tokenizer.decode(fp_tokens, skip_special_tokens=True),
                       'top2_margins': fp_margins},
              'int8': {'tokens': int_tokens, 'text': tokenizer.decode(int_tokens, skip_special_tokens=True),
                       'top2_margins': int_margins},
              'comparison': result,
              'projection_weight_errors': matrix_errors,
              'precision': 'FP32 embeddings, activations, norms, tied LM head; all 168 transformer projections per-row INT8 reconstructed back to FP32',
              'limitations': ['PC-only reconstructed INT8 weight test, not hardware INT8 integer kernels',
                              'Greedy decoding with full-prefix recomputation: not a KV-cache correctness test',
                              'Only one short chat input; matching response is not general language-quality proof']}
    Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print('PHASE79 EXACT TOKEN SEQUENCE MATCH', result['exact_match'], flush=True)
    print('PHASE79 MATCHING PREFIX TOKENS', result['matching_prefix_tokens'], flush=True)
    print('PHASE79 FIRST DIFFERENT POSITION', result['first_different_generated_position'], flush=True)
    print('PHASE79 REPORT', args.report, flush=True)
    print('PHASE79 GATE', 'PASS' if result['exact_match'] else 'FAIL', flush=True)
    return 0 if result['exact_match'] else 2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prompt', default='Hello')
    p.add_argument('--max-new-tokens', type=int, default=8)
    p.add_argument('--offline', action='store_true')
    p.add_argument('--report', default='phase79_multitoken_report.json')
    raise SystemExit(run(p.parse_args()))


if __name__ == '__main__':
    main()
