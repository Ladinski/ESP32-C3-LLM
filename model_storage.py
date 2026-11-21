"""Canonical immutable TinyStories INT4 storage image for ESP32-driven range reads.

Offsets are derived from the fixed Phase 19 firmware layout. No neural computations
are performed by the storage service. File can later be copied to an SD card.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from phase15_core import Export, D, F

VOCAB=50257
ROW=36
POS_COUNT=2048
LAYERS=8
LAYER_SIZE=29440
VOCAB_BASE=0
POS_BASE=VOCAB*ROW
LAYER_BASE=POS_BASE+POS_COUNT*ROW
HEAD_BASE=LAYER_BASE+LAYERS*LAYER_SIZE
TOTAL=HEAD_BASE+512

def emit_rows(e, f, name, shape):
    item=e.item(name)
    if item['encoding']!='int4_per_row' or item['shape']!=list(shape):
        raise RuntimeError(f'Unexpected tensor {name}')
    raw=e.raw(name)
    scales=(e.root/item['scale_file']).read_bytes()
    width=(shape[1]+1)//2
    if len(raw)!=shape[0]*width or len(scales)!=shape[0]*4:
        raise RuntimeError('Invalid INT4 tensor length '+name)
    s=np.frombuffer(scales,dtype='<f4')
    if not np.isfinite(s).all() or (s<=0).any():
        raise RuntimeError('Nonfinite/negative INT4 scale '+name)
    for r in range(shape[0]):
        f.write(raw[r*width:(r+1)*width]);f.write(scales[r*4:(r+1)*4])

def emit_fp(e,f,name,shape):
    f.write(e.fp(name,shape).astype('<f4').tobytes())

def pack(export_dir='phase13_export',output='phase19_model.bin'):
    e=Export(export_dir)
    c=e.config
    if (int(c['vocab_size'])!=VOCAB or int(c['max_position_embeddings'])!=POS_COUNT
        or int(c['num_layers'])!=LAYERS or int(c['hidden_size'])!=D
        or int(c['num_heads'])!=16 or list(c['attention_layers'])!=['global','local']*4
        or int(c.get('window_size',256))<16 or float(c.get('layer_norm_epsilon',1e-5))!=1e-5
        or 'lm_head.weight' in e.items):
        raise RuntimeError('Checkpoint configuration not compatible with Phase 19 layout')
    output=Path(output)
    with output.open('wb') as f:
        emit_rows(e,f,'transformer.wte.weight',(VOCAB,D))
        assert f.tell()==POS_BASE
        emit_rows(e,f,'transformer.wpe.weight',(POS_COUNT,D))
        assert f.tell()==LAYER_BASE
        for i in range(LAYERS):
            p=f'transformer.h.{i}.';a=p+'attn.attention.';m=p+'mlp.'
            start=f.tell()
            emit_fp(e,f,p+'ln_1.weight',(D,));emit_fp(e,f,p+'ln_1.bias',(D,))
            for pr in ('q_proj','k_proj','v_proj'): emit_rows(e,f,a+pr+'.weight',(D,D))
            emit_fp(e,f,a+'out_proj.bias',(D,));emit_rows(e,f,a+'out_proj.weight',(D,D))
            emit_fp(e,f,p+'ln_2.weight',(D,));emit_fp(e,f,p+'ln_2.bias',(D,))
            emit_fp(e,f,m+'c_fc.bias',(F,));emit_rows(e,f,m+'c_fc.weight',(F,D))
            emit_fp(e,f,m+'c_proj.bias',(D,));emit_rows(e,f,m+'c_proj.weight',(D,F))
            if f.tell()-start!=LAYER_SIZE:raise RuntimeError('Layer offset mismatch')
        assert f.tell()==HEAD_BASE
        emit_fp(e,f,'transformer.ln_f.weight',(D,))
        emit_fp(e,f,'transformer.ln_f.bias',(D,))
        assert f.tell()==TOTAL
    checksum=hashlib.sha256(output.read_bytes()).hexdigest()
    manifest={'format':'ESP32_C3_GPTNEO_PHASE19_V1','source':'roneneldan/TinyStories-1M','bytes':TOTAL,
              'sha256':checksum,'vocabulary_offset':VOCAB_BASE,'positions_offset':POS_BASE,
              'layers_offset':LAYER_BASE,'layer_bytes':LAYER_SIZE,'head_norm_offset':HEAD_BASE,
              'layout':'vocab INT4 rows | position INT4 rows | 8 layer packet groups | final LN FP32'}
    output.with_suffix('.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(f'PACK PASS: {output} | {TOTAL:,} bytes | SHA256 {checksum}')
    print('The file contains actual pretrained data and can later be copied to SD/microSD.')
    return manifest

def verify(output='phase19_model.bin'):
    output=Path(output)
    meta=json.loads(output.with_suffix('.json').read_text(encoding='utf-8'))
    if (meta['format']!='ESP32_C3_GPTNEO_PHASE19_V1' or meta['bytes']!=TOTAL
        or output.stat().st_size!=TOTAL): raise RuntimeError('Storage layout/length mismatch')
    hasher=hashlib.sha256()
    with output.open('rb') as f:
        for block in iter(lambda:f.read(65536),b''):hasher.update(block)
    if hasher.hexdigest()!=meta['sha256']:raise RuntimeError('Storage SHA256 mismatch')
    print(f'CHECK PASS: immutable storage image | {TOTAL:,} bytes | SHA256 {hasher.hexdigest()}')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--export-dir',default='phase13_export');ap.add_argument('--output',default='phase19_model.bin');ap.add_argument('--check',action='store_true')
    args=ap.parse_args();verify(args.output) if args.check else pack(args.export_dir,args.output)
