"""PC-ONLY comparison of original Qwen Instruct vs the separately exported full INT4 image.

The first-position, *plain-text* test is intentionally NOT an instruction/chat-quality test.
No ESP32 inference, no COM3 and no existing base-model files are changed.
"""
import argparse
import hashlib
import json
import mmap
from pathlib import Path

REPO='Qwen/Qwen2.5-0.5B-Instruct'


def verify(path, meta):
    p=Path(path)
    if p.stat().st_size!=meta['image_size']:
        raise ValueError(f'Wrong image length {p}: {p.stat().st_size} != {meta["image_size"]}')
    with p.open('rb') as f:
        value=hashlib.file_digest(f,'sha256').hexdigest()
    if value!=meta['sha256']:
        raise ValueError(f'SHA mismatch for {p}: {value}')
    if meta['model']!=REPO:
        raise ValueError(f'Wrong checkpoint label in {p}: {meta["model"]}')
    return value


def original_reference(checkpoint, token_id):
    import torch
    from transformers import AutoModelForCausalLM
    print('Loading original Instruct model on PC for fidelity comparison...',flush=True)
    model=AutoModelForCausalLM.from_pretrained(REPO,torch_dtype=torch.float32,attn_implementation='eager')
    model.eval().to('cpu')
    with torch.inference_mode():
        scores=model(input_ids=torch.tensor([[token_id]]),use_cache=False).logits[0,-1].float()
        top=torch.topk(scores,2)
        result=(int(top.indices[0]),float(top.values[0]),float(top.values[0]-top.values[1]))
    del model
    return result


def run(args):
    import numpy as np
    from tokenizers import Tokenizer
    from huggingface_hub import hf_hub_download
    from export_phase45_head import BFloatReader
    from phase69_reference_math import reference
    from phase69_full_reference import head_reference
    checkpoint=args.checkpoint or hf_hub_download(REPO,'model.safetensors')
    tokenizer_file=args.tokenizer or hf_hub_download(REPO,'tokenizer.json')
    tokenizer=Tokenizer.from_file(str(tokenizer_file))
    ids=tokenizer.encode(args.prompt,add_special_tokens=False).ids
    if len(ids)!=1:raise ValueError(f'This first fidelity gate requires exactly ONE raw prompt token, got {ids}; use --prompt Hello')
    token_id=ids[0]
    model_meta=json.loads(Path(args.metadata).read_text(encoding='utf8'))
    head_meta=json.loads(Path(args.head_metadata).read_text(encoding='utf8'))
    if len(model_meta['layers'])!=24 or head_meta['vocab']!=151936 or head_meta['hidden']!=896:
        raise ValueError('Expected 24 layers, 151936 vocabulary rows, 896 hidden dimensions')
    print('PHASE73 | INSTRUCT INT4 FIDELITY GATE | PC ONLY',flush=True)
    print('PROMPT',repr(args.prompt),'TOKEN IDs',ids,flush=True)
    print('TRANSFORMER SHA256',verify(args.image,model_meta),flush=True)
    print('HEAD SHA256',verify(args.head,head_meta),flush=True)
    with BFloatReader(checkpoint) as src:
        emb=np.asarray(src.get_row('model.embed_tokens.weight',token_id),dtype=np.float32).copy()
    with open(args.image,'rb') as f,open(args.head,'rb') as h:
        mm=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ)
        hm=mmap.mmap(h.fileno(),0,access=mmap.ACCESS_READ)
        try:
            states=reference(mm,model_meta['layers'],[emb])
            state=states[(0,23)]
            int4_id,int4_logit,int4_second=head_reference(hm,state,head_meta)
        finally:
            hm.close();mm.close()
    print('EXPORTED INT4 FIRST TOKEN',int4_id,repr(tokenizer.decode([int4_id])),
          'score',int4_logit,'margin',int4_logit-int4_second,flush=True)
    orig_id,orig_score,orig_margin=original_reference(checkpoint,token_id)
    print('ORIGINAL CHECKPOINT FIRST TOKEN',orig_id,repr(tokenizer.decode([orig_id])),
          'score',orig_score,'margin',orig_margin,flush=True)
    match=int4_id==orig_id
    print('PHASE73 FIRST-TOKEN FIDELITY', 'MATCH' if match else 'MISMATCH',flush=True)
    print('NOTE: This is plain, one-token text, NOT the Instruct chat template or conversational quality.',flush=True)
    report={'model':REPO,'prompt':args.prompt,'input_ids':ids,'source_checkpoint':str(checkpoint),
            'int4_token':int4_id,'original_token':orig_id,'match':match,
            'int4_logit':int4_logit,'original_logit':orig_score,'int4_margin':int4_logit-int4_second,
            'original_margin':orig_margin,'checks':'full 24-layer NumPy INT4 vs original float32 checkpoint, single raw input token'}
    Path(args.report).write_text(json.dumps(report,indent=2),encoding='utf8')
    print('REPORT',args.report,flush=True)
    if not match:
        print('PHASE73 GATE FAIL: Do not switch ESP32 model images. Investigate quantization fidelity first.',flush=True)
        return 2
    print('PHASE73 GATE PASS FOR THIS INPUT ONLY. Chat-template/long-context and ESP32 checks remain.',flush=True)
    return 0

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prompt',default='Hello')
    parser.add_argument('--checkpoint',help='Official local Instruct model.safetensors, otherwise download from HF cache')
    parser.add_argument('--tokenizer',help='Optional local official Instruct tokenizer.json')
    parser.add_argument('--image',default='phase25_qwen_24layers_int4.bin')
    parser.add_argument('--metadata',default='phase25_qwen_24layers_int4.json')
    parser.add_argument('--head',default='phase45_qwen_head_int4.bin')
    parser.add_argument('--head-metadata',default='phase45_qwen_head_int4.json')
    parser.add_argument('--report',default='phase73_instruct_fidelity_report.json')
    raise SystemExit(run(parser.parse_args()))

