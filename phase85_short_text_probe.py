"""Offline-only short-prompt next-token test. No ESP32 connection or inference offload."""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'
CANDIDATES = [
    ('Sky:', ' blue'),
    ('Sky color:', ' blue'),
    ('The sky is', ' blue'),
    ('France capital:', ' Paris'),
    ('Capital of France:', ' Paris'),
    ('UK capital:', ' London'),
]

print('OFFLINE REFERENCE ONLY: no ESP32 connected', flush=True)
print('Loading locally cached original model on CPU...', flush=True)
tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, torch_dtype=torch.float32, attn_implementation='eager')
model.eval().requires_grad_(False)

with torch.inference_mode():
    for prompt, desired in CANDIDATES:
        ids = tok(prompt, return_tensors='pt', add_special_tokens=False).input_ids
        logits = model(input_ids=ids, use_cache=False).logits[0, -1].float()
        scores, top_ids = torch.topk(logits, 5)
        actual_id = int(top_ids[0])
        actual = tok.decode([actual_id], clean_up_tokenization_spaces=False)
        desired_ids = tok(desired, add_special_tokens=False).input_ids
        rank = None
        if len(desired_ids) == 1:
            rank = int((logits > logits[desired_ids[0]]).sum().item()) + 1
        print(f'\nPROMPT {prompt!r} | input_tokens={ids.shape[1]} | IDs={ids[0].tolist()}')
        print(f'PREDICTED token={actual_id} text={actual!r}')
        print(f'DESIRED {desired!r} | token_ids={desired_ids} | rank={rank if rank is not None else "multi-token"}')
        print('TOP 5:', ' | '.join(f'{int(i)}:{tok.decode([int(i)], clean_up_tokenization_spaces=False)!r} ({float(v):.3f})' for v,i in zip(scores,top_ids)))
        print('ONE-TOKEN TARGET MATCH:', actual == desired)
print('\nThis checks the original float model, not your ESP32 INT8 export.',flush=True)
