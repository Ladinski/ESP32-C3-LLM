#!/usr/bin/env python3
"""Phase 21: bounded ESP32 paging with verifiable PC storage-only mode.

PC duties: GPT-2 tokenization, bytes-only file reads, independent verification.
ESP32 duties: all INT4 math, transformer layers, KV cache, logits, top-k.
"""
import argparse
import hashlib
import math
import time
from pathlib import Path
import numpy as np
from phase15_core import Export, D, fnv, layernorm
from run_pretrained_kv import block, prepare, LAYERS, MAX_CONTEXT, MAX_PREFILL, VOCAB
from model_storage import (verify as verify_image, ROW, POS_BASE, LAYER_BASE,
                           LAYER_SIZE, HEAD_BASE, TOTAL)

# Expected logical byte access order for verification; host never chooses device page requests.
LAYER_RECORD_LENGTHS = ([256,256]+[ROW]*192+[256]+[ROW]*64+[256,256]
                        +[1024]+[ROW]*256+[256]+[132]*64)
assert len(LAYER_RECORD_LENGTHS)==583
assert sum(LAYER_RECORD_LENGTHS)==LAYER_SIZE

def requests_for(ids,step):
    # ESP32 determines these addresses independently; the host checks them
    # as a protocol invariant, then serves exactly the requested bytes.
    offsets=[]
    positions = range(len(ids)) if step==1 else [len(ids)-1]
    for pos in positions:
        offsets.extend(((ids[pos]*ROW,ROW),(POS_BASE+pos*ROW,ROW)))
    for layer in range(LAYERS):
        offset=LAYER_BASE+layer*LAYER_SIZE
        for length in LAYER_RECORD_LENGTHS:
            offsets.append((offset,length));offset+=length
    offsets.extend(((HEAD_BASE,256),(HEAD_BASE+256,256)))
    offsets.extend((i*ROW,ROW) for i in range(VOCAB))
    assert len(offsets)==(2*len(ids) if step==1 else 2)+LAYERS*583+2+VOCAB
    return offsets

PAGE_SIZE = 512

def expected_pages(logical_requests):
    """Exact single-page-cache trace for validating device-selected page addresses."""
    page = None
    result = []
    cache_hits = 0
    for off, count in logical_requests:
        stop = off + count
        while off < stop:
            current = (off // PAGE_SIZE) * PAGE_SIZE
            if page == current:
                cache_hits += 1
            else:
                result.append((current, min(PAGE_SIZE, TOTAL-current)))
                page = current
            off = min(stop, current + PAGE_SIZE)
    return result, cache_hits

class StorageService:
    def __init__(self,model_path,expected):
        self.f=open(model_path,'rb')
        self.expected=expected
        self.pages, self.expected_hits=expected_pages(expected)
        self.index=0
        self.h=2166136261
        self.size=0
        self.requests=0
        self.logical_size=sum(n for _,n in expected)
        self.logical_hash=2166136261
        self._verify_logical_hash(model_path)
        self.start=time.monotonic()
        self.memory_samples=[]
    def _verify_logical_hash(self, model_path):
        # Independent reference of the original logical stream from Phase 19.
        # Reads use a separate file handle, so they cannot influence the server cursor.
        with open(model_path, 'rb') as verify_file:
            for off, length in self.expected:
                verify_file.seek(off)
                data = verify_file.read(length)
                if len(data) != length: raise RuntimeError('Truncated model during logical hash check')
                for b in data: self.logical_hash=((self.logical_hash ^ b)*16777619)&0xffffffff
    def close(self):self.f.close()
    def handle(self,line,port):
        parts=line.split()
        if len(parts)!=3 or parts[0]!='PAGE':raise RuntimeError('Malformed page request '+line)
        off,size=map(int,parts[1:]);i=self.index
        if i>=len(self.pages):raise RuntimeError('Unexpected extra storage PAGE: '+line)
        expect_off,expect_len=self.pages[i]
        if (off,size)!=(expect_off,expect_len):
            raise RuntimeError(f'Page request {i+1} mismatch: actual {off}/{size}, expected {expect_off}/{expect_len}')
        if not 0<size<=PAGE_SIZE or off%PAGE_SIZE or not 0<=off<=TOTAL-size:raise RuntimeError('Invalid storage address')
        self.f.seek(off);payload=self.f.read(size)
        if len(payload)!=size:raise RuntimeError('Truncated model image')
        port.write(payload)
        for b in payload:self.h=((self.h^b)*16777619)&0xffffffff
        self.size+=size;self.index+=1;self.requests+=1
        if self.requests%1024==0:print(f'  ESP32 fetched {self.requests:,}/{len(self.pages):,} model pages',flush=True)
    def expect(self,port,prefix,timeout=60):
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            line=port.readline().decode('ascii',errors='replace').strip()
            if not line:continue
            if line.startswith('PAGE '):self.handle(line,port);deadline=time.monotonic()+timeout;continue
            if line=='ACK':raise RuntimeError('Old firmware: unexpected per-record ACK')
            if line.startswith('STAGE '):print('  ESP32:',line,flush=True);continue
            if line.startswith('MEM '):
                fields=line.split()
                if len(fields)!=7:raise RuntimeError('Malformed memory metrics: '+line)
                tag=fields[1]
                free,minfree,stack,statics,lowest=map(int,fields[2:])
                if min(free,minfree,stack,lowest)<=0 or free<65536 or minfree<65536 or statics>163840:
                    raise RuntimeError('Memory-budget violation: '+line)
                self.memory_samples.append((tag,free,minfree,stack,statics,lowest))
                print('  ESP32 memory:',line,flush=True)
                continue
            if line.startswith('ERR '):raise RuntimeError('ESP32 firmware: '+line)
            if line.startswith(prefix):return line
            if line.startswith('ESP32-C3 PHASE 21'):continue
            raise RuntimeError(f'Expected {prefix!r}; received {line!r}')
        raise TimeoutError(f'Timeout waiting for {prefix!r}, served {self.index}/{len(self.pages)} storage pages')

def run_step(e,prepared,model,ids,step,port,verify_every_layer=True,storage_only=False):
    n=len(ids)
    refs=[]
    expected_top=None
    logits=None
    if not storage_only:
        wte,wpe,layers,lnw,lnb,head,positions,vocab_rows=prepared
        x=(wte[np.asarray(ids)]+wpe[:n]).astype(np.float32)
        for l in range(LAYERS):
            x=block(e,x,l)
            if not np.isfinite(x).all():raise RuntimeError('Nonfinite Python reference')
            refs.append(x.copy())
        normalized=layernorm(x,lnw,lnb,1e-5)
        logits=wte @ normalized[-1]
        expected_top=sorted(range(VOCAB),key=lambda i:(-float(logits[i]),i))[:5]
    offsets=requests_for(ids,step)
    pages, cache_hits = expected_pages(offsets)
    print(f'Step {step}: context={n}, logical weight records={len(offsets):,}, expected page fetches={len(pages):,}, cached page uses={cache_hits:,}',flush=True)
    if port is None:
        print('  Python reference top-5:',expected_top)
        return expected_top[0]
    svc=StorageService(model,offsets)
    try:
        port.write(b'HELLO\n');print(' ',svc.expect(port,'READY PHASE21'))
        typ='NEW' if storage_only or e.activation=='gelu_new' else 'EXACT'
        command=f'PREFILL {typ} {n} '+ ' '.join(map(str,ids)) if step==1 else f'DECODE {typ} {ids[-1]}'
        port.write((command+'\n').encode('ascii'))
        print(' ',svc.expect(port,'GO PHASE21'))
        for l in range(LAYERS):
            if l:
                port.write(f'NEXT {l}\n'.encode('ascii'))
                svc.expect(port,f'GO LAYER {l}')
            actual=np.empty((n if step==1 else 1,D),dtype=np.float32)
            for t in range(len(actual)):
                for chunk in range(4):
                    line=svc.expect(port,f'OUT {l} {t} {chunk} ')
                    a=line.split()
                    if len(a)!=20 or list(map(int,a[1:4]))!=[l,t,chunk]:
                        raise RuntimeError('Unexpected layer output: '+line)
                    actual[t,chunk*16:(chunk+1)*16]=[float(z) for z in a[4:]]
            svc.expect(port,f'LAYER_DONE {l}')
            if not np.isfinite(actual).all():raise RuntimeError(f'Layer {l} nonfinite ESP32 output')
            if not storage_only:
                err=float(np.max(np.abs(actual-(refs[l] if step==1 else refs[l][-1:]))))
                if verify_every_layer:print(f'  Layer {l} max error {err:.3g}',flush=True)
                if err>0.02:raise RuntimeError(f'Layer {l} error {err}')
        svc.expect(port,'LAYERS_DONE')
        port.write(b'HEAD 50257\n');svc.expect(port,'GO HEAD')
        print('  ESP32 requesting/scoring 50,257 vocab rows...',flush=True)
        tops=[]
        for pos in range(5):
            a=svc.expect(port,f'TOP {pos} ').split()
            if len(a)!=4 or int(a[1])!=pos:raise RuntimeError('Bad TOP response '+str(a))
            tops.append((int(a[2]),float(a[3])))
        done=svc.expect(port,'DONE ').split()
        if len(done)!=10:raise RuntimeError('Bad DONE fields '+str(done))
        if not svc.memory_samples or not any(t[0]=='BEGIN' for t in svc.memory_samples) or not any(t[0]=='HEAD' for t in svc.memory_samples) or sum(t[0]=='LAYER' for t in svc.memory_samples)!=LAYERS:
            raise RuntimeError('Missing device memory telemetry')
        if (int(done[1])!=svc.logical_size or int(done[2],16)!=svc.logical_hash
            or int(done[7])!=svc.requests or int(done[8])!=svc.size
            or int(done[9])!=svc.expected_hits or svc.index!=len(svc.pages)):
            raise RuntimeError('Storage request count/bytes/hash mismatch '+str(done))
        if storage_only:
            if len({i for i,_ in tops}) != 5 or any(i<0 or i>=VOCAB or not math.isfinite(v) for i,v in tops):
                raise RuntimeError('Invalid device top-5')
            if any(tops[i][1] < tops[i+1][1] for i in range(4)):
                raise RuntimeError('Device top-5 not ordered')
        else:
            if [i for i,_ in tops]!=expected_top:raise RuntimeError('Top-5 disagreement')
            err=max(abs(score-float(logits[i])) for i,score in tops)
            if err>0.02:raise RuntimeError('Logit mismatch '+str(err))
        port.write(b'CACHE STATUS\n')
        expected_cache=f'CACHE {n}/{MAX_CONTEXT}'
        if svc.expect(port,'CACHE ')!=expected_cache:raise RuntimeError('KV cache position mismatch')
        print(f'  Storage PASS: {len(offsets):,} logical records, {svc.requests:,} ESP32 page fetches, {svc.size:,} physical bytes')
        print(f'  Logical stream: {svc.logical_size:,} bytes, FNV32 {svc.logical_hash:08x} | page-cache hits: {svc.expected_hits:,}')
        if storage_only:print(f'  ESP32 top-5 {[i for i,_ in tops]} | NO HOST MODEL INFERENCE',flush=True)
        else:print(f'  ESP32 top-5 {[i for i,_ in tops]} | max logit error {err:.7g}')
        print(f'  Device time {int(done[6])/1e6:.3f}s | host time {time.monotonic()-svc.start:.3f}s')
        return tops[0][0]
    finally:svc.close()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--export-dir',default='phase13_export')
    ap.add_argument('--model',default='phase19_model.bin')
    ap.add_argument('--port',default='COM3')
    ap.add_argument('--prompt',default='Once upon a time')
    ap.add_argument('--max-new-tokens',type=int,default=3)
    ap.add_argument('--self-test',action='store_true')
    ap.add_argument('--storage-only',action='store_true',help='No PC-side transformer/head inference, only tokenizer and raw page service')
    args=ap.parse_args()
    if not 1<=args.max_new_tokens<=5:ap.error('Choose 1 to 5 new tokens')
    if args.storage_only and args.self_test:ap.error('--storage-only and --self-test cannot be combined')
    print('PHASE 21 | bounded ESP32 512-byte page cache')
    verify_image(args.model)
    e=None if args.storage_only else Export(args.export_dir)
    from transformers import GPT2TokenizerFast
    tokenizer=GPT2TokenizerFast.from_pretrained('gpt2')
    ids=tokenizer.encode(args.prompt,add_special_tokens=False)
    if not 1<=len(ids)<=MAX_PREFILL or len(ids)+args.max_new_tokens-1>MAX_CONTEXT:
        raise RuntimeError('Prompt exceeds Phase 19 context limits')
    print('Prompt:',repr(args.prompt),'| token IDs:',ids)
    prepared=None if args.storage_only else prepare(e)
    if args.storage_only:print("STORAGE-ONLY MODE: no model tensors loaded or neural inference computed by Python", flush=True)
    gen=[]
    if args.self_test:
        for step in range(1,args.max_new_tokens+1):
            new=run_step(e,prepared,args.model,ids,step,None)
            gen.append(new);ids.append(new)
        print('PHASE 21 REFERENCE PASS (no hardware test)')
        return
    import serial
    with serial.Serial(args.port,115200,timeout=0.2,write_timeout=10) as port:
        port.reset_input_buffer()
        for step in range(1,args.max_new_tokens+1):
            new=run_step(e,prepared,args.model,ids,step,port,storage_only=args.storage_only)
            gen.append(new);ids.append(new)
            print(f'TOKEN {step}: {new}, text={tokenizer.decode(ids)!r}',flush=True)
            if new==tokenizer.eos_token_id:break
    print('PHASE 21 PASS: bounded ESP32 paging + pretrained KV decoding')
    print('Generated IDs:',gen,'Text:',repr(tokenizer.decode(ids)))
    print('PC role: immutable byte-range server and independent post-hoc numerical reference only.')
    print('SD/microSD can replace the read-only PAGE server in a later phase.')

if __name__=='__main__':main()
