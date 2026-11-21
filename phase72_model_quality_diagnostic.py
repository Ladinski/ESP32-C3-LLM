"""Phase 72: Compare original Qwen2.5 base and Instruct models on PC.

This is intentionally a PC-only diagnosis. It is NOT ESP32 inference and does not
validate the existing on-device INT4 weight image. Neither model is re-quantized.
"""
import argparse
import gc
import json
from pathlib import Path

BASE_REPO = 'Qwen/Qwen2.5-0.5B'
INSTRUCT_REPO = 'Qwen/Qwen2.5-0.5B-Instruct'
EXPECTED_ESP32_FIRST_TOKEN_FOR_HELLO = 52643


def make_inputs(tokenizer, repo, prompt):
    if repo == INSTRUCT_REPO:
        messages = [{'role': 'user', 'content': prompt}]
        token_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        if not token_ids or not isinstance(token_ids, list):
            raise RuntimeError('Failed to apply Instruct chat template')
        return token_ids
    return tokenizer.encode(prompt, add_special_tokens=False)


def run_one(repo, prompt, new_tokens, device):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f'\nPHASE72 LOADING {repo}', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(repo, use_fast=True)
    ids = make_inputs(tokenizer, repo, prompt)
    print('INPUT TOKEN IDS', ids, flush=True)
    if not ids:
        raise ValueError('Prompt tokenized to empty sequence')

    # FP32 on CPU gives a stable, transparent checkpoint diagnostic.
    # On CUDA, BF16/FP16 usage can be configured later; correctness first.
    model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float32)
    model.eval()
    model.to(device)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        # The original checkpoint's greedy prediction on the full prompt.
        out = model(input_ids=input_ids, attention_mask=mask, use_cache=False)
        scores = out.logits[0, -1].float()
        top = torch.topk(scores, 2)
        first_id = int(top.indices[0].item())
        first_score = float(top.values[0].item())
        margin = float((top.values[0] - top.values[1]).item())
        del out, scores, top
        produced = model.generate(
            input_ids=input_ids,
            attention_mask=mask,
            max_new_tokens=new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )[0, len(ids):].tolist()
    if not produced or produced[0] != first_id:
        raise RuntimeError('First greedy token from generate() differs from independent forward pass')
    response = tokenizer.decode(produced, skip_special_tokens=False)
    print('GREEDY FIRST TOKEN', first_id, repr(tokenizer.decode([first_id], skip_special_tokens=False)), flush=True)
    print('GREEDY FIRST LOGIT', first_score, 'TOP2 MARGIN', margin, flush=True)
    print('GENERATED IDS', produced, flush=True)
    print('GENERATED TEXT', repr(response), flush=True)
    result = {
        'repo': repo,
        'prompt': prompt,
        'prompt_token_ids': ids,
        'greedy_first_token_id': first_id,
        'greedy_first_logit': first_score,
        'first_top2_margin': margin,
        'generated_token_ids': produced,
        'generated_text': response,
        'pc_side_neural_inference': True,
        'device': device,
        'checkpoint_precision_loaded': 'float32 (source checkpoint weights)',
    }
    del input_ids, mask, model, tokenizer
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prompt', default='Hello')
    parser.add_argument('--new-tokens', type=int, default=2)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--only-base', action='store_true', help='Skip downloading the Instruct checkpoint')
    parser.add_argument('--report', default='phase72_model_quality_report.json')
    args = parser.parse_args()
    if not 1 <= args.new_tokens <= 16:
        parser.error('--new-tokens must be between 1 and 16')
    if args.device == 'cuda':
        import torch
        if not torch.cuda.is_available():
            parser.error('CUDA is not available; use --device cpu')
    repos = [BASE_REPO] if args.only_base else [BASE_REPO, INSTRUCT_REPO]
    results = []
    for repo in repos:
        results.append(run_one(repo, args.prompt, args.new_tokens, args.device))
    base = results[0]
    if args.prompt == 'Hello':
        match = base['greedy_first_token_id'] == EXPECTED_ESP32_FIRST_TOKEN_FOR_HELLO
        print('\nPHASE72 HELLO ORIGINAL BASE vs ESP32 INT4 FIRST TOKEN:',
              'MATCH' if match else 'DIFFERENT', flush=True)
        print('PC ORIGINAL BASE', base['greedy_first_token_id'],
              '| ESP32 INT4', EXPECTED_ESP32_FIRST_TOKEN_FOR_HELLO, flush=True)
        if not match:
            print('INVESTIGATE: This difference could reflect INT4 quantization, implementation, or both.', flush=True)
        else:
            print('OBSERVATION: Both select the same first token on Hello; this does not prove equal logits or full accuracy.', flush=True)
    if len(results) > 1:
        print('PHASE72 INSTRUCT RESPONSE:', repr(results[1]['generated_text']), flush=True)
        print('IMPORTANT: This is a DIFFERENT checkpoint; the existing ESP32 INT4 weights remain the BASE model.', flush=True)
    report = {
        'type': 'PC-only model quality diagnostic',
        'esp32_first_token_for_hello': EXPECTED_ESP32_FIRST_TOKEN_FOR_HELLO,
        'results': results,
    }
    Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print('PHASE72 REPORT', args.report, flush=True)
    print('PHASE72 DIAGNOSTIC COMPLETE: no ESP32 firmware change, flash, or device neural inference.', flush=True)


if __name__ == '__main__':
    main()
