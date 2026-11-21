"""Export real pretrained Qwen2.5-0.5B final norm and lm_head to a SEPARATE per-row INT4 image.

This is an offline conversion. It performs no neural inference on the PC.
"""
import argparse, hashlib, json, struct, mmap
from pathlib import Path
import numpy as np
REPO='Qwen/Qwen2.5-0.5B-Instruct'
D=896

class BFloatReader:
    """Read one safetensors tensor or row at a time without allocating checkpoint-sized RAM."""
    def __init__(self,path):
        self.file=open(path,'rb');self.mm=mmap.mmap(self.file.fileno(),0,access=mmap.ACCESS_READ)
        length=struct.unpack_from('<Q',self.mm,0)[0]
        if length>10_000_000:raise ValueError('Unexpected safetensors header length')
        self.meta=json.loads(self.mm[8:8+length]);self.data_begin=8+length
    def __enter__(self):return self
    def __exit__(self,*args):self.mm.close();self.file.close()
    def decode(self,raw,dtype,shape):
        if dtype=='BF16':return (np.frombuffer(raw,dtype='<u2').astype(np.uint32)<<16).view(np.float32).reshape(shape)
        if dtype=='F32':return np.frombuffer(raw,dtype='<f4').copy().reshape(shape)
        if dtype=='F16':return np.frombuffer(raw,dtype='<f2').astype(np.float32).reshape(shape)
        raise ValueError('Unsupported pretrained tensor dtype '+dtype)
    def get_tensor(self,name):
        meta=self.meta[name];start,stop=meta['data_offsets']
        return self.decode(self.mm[self.data_begin+start:self.data_begin+stop],meta['dtype'],meta['shape'])
    def get_row(self,name,index):
        meta=self.meta[name];rows,cols=meta['shape']
        if not 0<=index<rows:raise IndexError(index)
        size={'BF16':2,'F16':2,'F32':4}[meta['dtype']]*cols
        start=self.data_begin+meta['data_offsets'][0]+index*size
        return self.decode(self.mm[start:start+size],meta['dtype'],(cols,))

VOCAB=151936
ROW_BYTES=D//2+4

def export(checkpoint, dest):
    path=Path(dest)
    meta_path=path.with_suffix('.json')
    header_path=Path('main/phase45_head_layout.h')
    with BFloatReader(checkpoint) as source:
        tensors=source.meta
        norm_key='model.norm.weight'
        if norm_key not in tensors:
            raise RuntimeError('Official checkpoint is missing model.norm.weight')
        if 'lm_head.weight' in tensors:
            head_key='lm_head.weight';source_type='untied lm_head.weight'
        elif 'model.embed_tokens.weight' in tensors:
            head_key='model.embed_tokens.weight';source_type='tied embedding weights'
        else:
            raise RuntimeError('Cannot locate genuine pretrained output projection')
        shape=tensors[head_key]['shape']
        if shape != [VOCAB,D]:
            raise RuntimeError(f'Unexpected output head dimensions: {shape}, expected {[VOCAB,D]}')
        norm=source.get_tensor(norm_key)
        if norm.shape!=(D,) or not np.isfinite(norm).all():
            raise RuntimeError('Invalid checkpoint final normalization tensor')
        with path.open('wb') as out:
            out.write(np.asarray(norm,dtype='<f4').tobytes())
            for index in range(VOCAB):
                row=source.get_row(head_key,index)
                if not np.isfinite(row).all():
                    raise RuntimeError(f'Nonfinite head row {index}')
                peak=float(np.max(np.abs(row)))
                scale=np.float32(peak/7.0 if peak else 1.0)
                q=np.clip(np.rint(row/scale),-8,7).astype(np.int8)
                packed=(q[::2].astype(np.uint8)&15)|((q[1::2].astype(np.uint8)&15)<<4)
                out.write(packed.tobytes());out.write(struct.pack('<f',float(scale)))
                if (index+1)%20000==0:
                    print(f'Exported {index+1:,}/{VOCAB:,} genuine pretrained output rows',flush=True)
    with path.open('rb') as stream: checksum=hashlib.file_digest(stream,'sha256').hexdigest()
    total=D*4+VOCAB*ROW_BYTES
    if path.stat().st_size!=total:raise RuntimeError('Head image size mismatch')
    info={'model':REPO,'source':source_type,'source_tensor':head_key,'quantization':'symmetric per-output-row INT4 with FP32 scale','vocab':VOCAB,'hidden':D,'final_norm_bytes':D*4,'row_bytes':ROW_BYTES,'head_offset':D*4,'image_size':total,'sha256':checksum}
    meta_path.write_text(json.dumps(info,indent=2),encoding='utf-8')
    header_path.parent.mkdir(exist_ok=True)
    header_path.write_text('''// Generated from official Qwen2.5-0.5B pretrained checkpoint.\n#pragma once\n#define P45_VOCAB %dU\n#define P45_HEAD_IMAGE_SIZE %dU\n#define P45_HEAD_ROW_BYTES %dU\n#define P45_HEAD_OFFSET %dU\n'''%(VOCAB,total,ROW_BYTES,D*4),encoding='utf-8')
    print(f'PHASE45 EXPORT PASS: {path} {total:,} bytes, source={source_type} SHA256 {checksum}',flush=True)
    print(f'Wrote {meta_path} and {header_path}',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',default=None,help='Local official model.safetensors; reuse HF cache if omitted')
    p.add_argument('--output',default='phase45_qwen_head_int4.bin')
    a=p.parse_args()
    if a.checkpoint:checkpoint=a.checkpoint
    else:
        from huggingface_hub import hf_hub_download
        checkpoint=hf_hub_download(REPO,'model.safetensors')
    export(checkpoint,a.output)
