#!/usr/bin/env python3
"""Phase 18: ESP32-C3 pretrained GPT-Neo greedy generation with persistent KV cache.

PC does GPT-2 BPE tokenization, weight selection/streaming and numerical
verification only. All forward-pass math and token ranking run on ESP32-C3.
The ESP32 retains K/V across decoding steps. The host independently recomputes a
full-sequence reference for strict numerical verification only.
"""
import argparse
import math
import time
import numpy as np
from phase15_core import Export, D, H, HD, F, fnv, layernorm, expect

LAYERS = 8
MAX_CONTEXT = 16
MAX_PREFILL = 8
VOCAB = 50257


def block(e, x, index):
    n = len(x)
    p = f'transformer.h.{index}.'
    attn = p + 'attn.attention.'
    mlp = p + 'mlp.'
    eps = float(e.config.get('layer_norm_epsilon', 1e-5))
    z = layernorm(x, e.fp(p+'ln_1.weight', (D,)), e.fp(p+'ln_1.bias', (D,)), eps)
    q = z @ e.matrix(attn+'q_proj.weight', (D,D)).T
    k = z @ e.matrix(attn+'k_proj.weight', (D,D)).T
    v = z @ e.matrix(attn+'v_proj.weight', (D,D)).T
    ctx = np.empty((n,D), dtype=np.float32)
    window = int(e.config.get('window_size', 256))
    local = e.config['attention_layers'][index] == 'local'
    for t in range(n):
        begin = max(0, t-window+1) if local else 0
        for h in range(H):
            a = h*HD
            scores = np.asarray([np.dot(q[t,a:a+HD], k[s,a:a+HD])/math.sqrt(HD)
                                 for s in range(begin,t+1)],dtype=np.float32)
            probs = np.exp(scores - scores.max()); probs /= probs.sum()
            ctx[t,a:a+HD] = probs @ v[begin:t+1,a:a+HD]
    x = x + ctx @ e.matrix(attn+'out_proj.weight',(D,D)).T + e.fp(attn+'out_proj.bias',(D,))
    z = layernorm(x,e.fp(p+'ln_2.weight',(D,)),e.fp(p+'ln_2.bias',(D,)),eps)
    a = z @ e.matrix(mlp+'c_fc.weight',(F,D)).T + e.fp(mlp+'c_fc.bias',(F,))
    if e.activation == 'gelu_new':
        a = 0.5*a*(1+np.tanh(np.sqrt(2/np.pi)*(a+0.044715*a**3)))
    else:
        a = 0.5*a*(1+np.vectorize(math.erf,otypes=[np.float64])(a/math.sqrt(2)).astype(np.float32))
    return np.asarray(x + a @ e.matrix(mlp+'c_proj.weight',(D,F)).T + e.fp(mlp+'c_proj.bias',(D,)),dtype=np.float32)


def layer_packets(e, index):
    p=f'transformer.h.{index}.'
    attn=p+'attn.attention.'
    mlp=p+'mlp.'
    out=[]
    def fp(name, shape): out.append(e.fp(name,shape).astype('<f4').tobytes())
    def mat(name,shape): out.extend(e.packets_for_matrix(name,shape))
    fp(p+'ln_1.weight',(D,)); fp(p+'ln_1.bias',(D,))
    for part in ('q_proj','k_proj','v_proj'): mat(attn+part+'.weight',(D,D))
    fp(attn+'out_proj.bias',(D,)); mat(attn+'out_proj.weight',(D,D))
    fp(p+'ln_2.weight',(D,)); fp(p+'ln_2.bias',(D,))
    fp(mlp+'c_fc.bias',(F,)); mat(mlp+'c_fc.weight',(F,D))
    fp(mlp+'c_proj.bias',(D,)); mat(mlp+'c_proj.weight',(D,F))
    if len(out)!=583: raise RuntimeError('Layer packet mismatch')
    return out


def prepare(e):
    cfg=e.config
    if (int(cfg['num_layers'])!=LAYERS or int(cfg['vocab_size'])!=VOCAB or
        list(cfg['attention_layers'])!=['global','local']*4 or
        float(cfg.get('layer_norm_epsilon',1e-5))!=1e-5 or
        int(cfg.get('window_size',256))<MAX_CONTEXT):
        raise RuntimeError('Unsupported model dimensions, LN epsilon or attention configuration')
    if 'lm_head.weight' in e.items or 'lm_head.bias' in e.items:
        raise RuntimeError('Output head not tied to token embedding: must adapt before inference')
    wte=e.matrix('transformer.wte.weight',(VOCAB,D))
    wpe=e.matrix('transformer.wpe.weight',(int(cfg['max_position_embeddings']),D))
    layers=[layer_packets(e,i) for i in range(LAYERS)]
    lnw=e.fp('transformer.ln_f.weight',(D,))
    lnb=e.fp('transformer.ln_f.bias',(D,))
    item=e.item('transformer.wte.weight')
    raw=e.raw('transformer.wte.weight')
    scales=(e.root/item['scale_file']).read_bytes()
    if len(raw)!=VOCAB*32 or len(scales)!=VOCAB*4:
        raise RuntimeError('Invalid embedding payload')
    vocab_rows=[raw[i*32:(i+1)*32]+scales[i*4:(i+1)*4] for i in range(VOCAB)]
    head=[lnw.astype('<f4').tobytes(),lnb.astype('<f4').tobytes()]+vocab_rows
    positions=[e.row_packet('transformer.wpe.weight',i) for i in range(MAX_CONTEXT)]
    return wte,wpe,layers,lnw,lnb,head,positions,vocab_rows


def run_step(e, prepared, ids, port, verify=True, step_no=1):
    wte,wpe,layers,lnw,lnb,head,positions,vocab_rows=prepared
    n=len(ids)
    x=(wte[np.asarray(ids)]+wpe[:n]).astype(np.float32)
    refs=[]
    for l in range(LAYERS):
        x=block(e,x,l)
        if not np.isfinite(x).all(): raise RuntimeError(f'Nonfinite layer {l}')
        refs.append(x.copy())
    normalized=layernorm(x,lnw,lnb,1e-5)
    logits=wte @ normalized[-1]
    expected=sorted(range(VOCAB),key=lambda i:(-float(logits[i]),i))[:5]
    decode = step_no > 1
    first=[]
    token_range = range(n-1,n) if decode else range(n)
    for t in token_range:
        first.extend((vocab_rows[ids[t]],positions[t]))
    all_packets=first+[packet for layer in layers for packet in layer]+head
    expected_bytes=sum(map(len,all_packets))
    expected_hash=fnv(b''.join(all_packets))
    print(f'Step {step_no} | {n} context tokens | mode={"DECODE" if decode else "PREFILL"} | {len(all_packets):,} records | '
          f'{expected_bytes:,} bytes | FNV32 {expected_hash:08x}',flush=True)
    if port is None:
        print(f'  PC reference greedy next ID: {expected[0]} | top-5 {expected}')
        return expected[0]
    port.write(b'HELLO\n')
    print(' ',expect(port,'READY PHASE18'))
    op = 'DECODE' if decode else 'PREFILL'
    command = f'{op} {"EXACT" if e.activation=="gelu" else "NEW"}'
    if not decode: command += f' {n}'
    port.write((command+'\n').encode('ascii'))
    print(' ',expect(port,'GO PHASE18'))
    started=time.perf_counter()
    def stream(records, label, interval=0):
        for index,payload in enumerate(records,1):
            port.write(payload)
            expect(port,'ACK',timeout_s=10,context=f'step {step_no}, {label} {index}/{len(records)}')
            if interval and (index%interval==0 or index==len(records)):
                print(f'  {label}: {index:,}/{len(records):,}',flush=True)
    stream(first,'embeddings')
    for l in range(LAYERS):
        if l:
            port.write(f'NEXT {l}\n'.encode('ascii'))
            expect(port,f'GO LAYER {l}',context=f'layer {l} start')
        stream(layers[l],f'layer {l}')
        n_out = 1 if decode else n
        actual=np.empty((n_out,D),dtype=np.float32)
        for t in range(n_out):
            for chunk in range(4):
                line=expect(port,f'OUT {l} {t} {chunk} ',timeout_s=30,
                            context=f'layer {l} token {t} chunk {chunk}')
                fields=line.split()
                if len(fields)!=20 or fields[1:4]!=[str(l),str(t),str(chunk)]:
                    raise RuntimeError('Malformed layer output: '+line)
                actual[t,chunk*16:(chunk+1)*16]=np.asarray([float(z) for z in fields[4:]],dtype=np.float32)
        expect(port,f'LAYER_DONE {l}',context=f'layer {l} complete')
        reference = refs[l][-1:] if decode else refs[l]
        err=float(np.max(np.abs(actual-reference)))
        if verify or l==LAYERS-1:
            print(f'  Layer {l}: max abs error {err:.3g}',flush=True)
        if not np.isfinite(actual).all() or err>0.02:
            raise RuntimeError(f'Layer {l} mismatch, error={err}')
    expect(port,'LAYERS_DONE',timeout_s=30)
    port.write(b'HEAD 50257\n')
    expect(port,'GO HEAD',timeout_s=30)
    print('  ESP32 computing full 50,257-token output head...',flush=True)
    stream(head,'vocabulary head',4096)
    tops=[]
    for pos in range(5):
        line=expect(port,f'TOP {pos} ',timeout_s=30,context='top-5 result')
        parts=line.split()
        if len(parts)!=4 or int(parts[1])!=pos:
            raise RuntimeError('Malformed top-5: '+line)
        tops.append((int(parts[2]),float(parts[3])))
    done=expect(port,'DONE ',timeout_s=30,context='final done')
    fields=done.split()
    if len(fields)!=7 or int(fields[1])!=expected_bytes or int(fields[2],16)!=expected_hash:
        raise RuntimeError('Transport byte count/hash mismatch: '+done)
    if [i for i,_ in tops]!=expected:
        raise RuntimeError(f'ESP32 top-5 {tops} differs from Python {expected}')
    err=max(abs(score-float(logits[token_id])) for token_id,score in tops)
    if err>0.02: raise RuntimeError(f'Logit error {err} exceeds threshold')
    port.write(b'CACHE STATUS\n')
    cache_line=expect(port,'CACHE ',timeout_s=10,context='post-inference cache status')
    if cache_line != f'CACHE {n}/{MAX_CONTEXT}':
        raise RuntimeError(f'Persistent KV cache length mismatch: {cache_line}; expected {n}')
    print(f'  Device KV cache: {cache_line}')
    print(f'  Device top-5: {[i for i,_ in tops]} | max logit error {err:.7g}')
    print(f'  Device time {int(fields[6])/1e6:.3f}s | host time {time.perf_counter()-started:.3f}s')
    return tops[0][0]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--export-dir',default='phase13_export')
    ap.add_argument('--port',default='COM3')
    ap.add_argument('--prompt',default='Once upon a time')
    ap.add_argument('--max-new-tokens',type=int,default=3)
    ap.add_argument('--self-test',action='store_true')
    ap.add_argument('--token-ids',type=int,nargs='+',help='Optional explicit GPT-2 token IDs instead of --prompt')
    args=ap.parse_args()
    if not 1<=args.max_new_tokens<=5:
        ap.error('--max-new-tokens must be between 1 and 5 (Phase 18 scope)')
    e=Export(args.export_dir)
    print('ESP32-C3 PHASE 18 | Pretrained generation with ESP32-resident KV cache')
    print('NOTE: PC handles GPT-2 tokenization and supplies weights; ESP32 computes every prediction.')
    from transformers import GPT2TokenizerFast
    tokenizer=GPT2TokenizerFast.from_pretrained('gpt2')
    ids=list(args.token_ids) if args.token_ids is not None else tokenizer.encode(args.prompt,add_special_tokens=False)
    if not ids or len(ids)>MAX_PREFILL or len(ids)+args.max_new_tokens-1>MAX_CONTEXT:
        raise RuntimeError(f'Prompt has {len(ids)} tokens. Need 1..{MAX_PREFILL} prefill tokens and total <= {MAX_CONTEXT}. '
                           'Shorten --prompt or reduce --max-new-tokens.')
    if any(i<0 or i>=VOCAB for i in ids): raise RuntimeError('Invalid GPT-2 token ID')
    print('Prompt:',repr(tokenizer.decode(ids)), '| token IDs:',ids)
    prepared=prepare(e)
    generated=[]
    if args.self_test:
        for i in range(args.max_new_tokens):
            next_id=run_step(e,prepared,ids,None,step_no=i+1)
            generated.append(next_id)
            ids.append(next_id)
            print('  Reference text:',repr(tokenizer.decode(ids)))
        print('PHASE 18 PYTHON REFERENCE PASS (no ESP32 run)')
        return
    import serial
    with serial.Serial(args.port,115200,timeout=0.3,write_timeout=10) as port:
        port.reset_input_buffer()
        for i in range(args.max_new_tokens):
            next_id=run_step(e,prepared,ids,port,step_no=i+1)
            generated.append(next_id)
            ids.append(next_id)
            print(f'TOKEN {i+1}: ID {next_id}, decoded piece {tokenizer.decode([next_id])!r}',flush=True)
            print('TEXT:',repr(tokenizer.decode(ids)),flush=True)
            if next_id==tokenizer.eos_token_id:
                print('EOS emitted; stopping generation.')
                break
    print('PHASE 18 PASS: pretrained autoregressive decoding with persistent ESP32 KV cache')
    print('Generated token IDs:',generated)
    print('Complete text:',repr(tokenizer.decode(ids)))
    print('KV CACHE: previous per-layer attention keys/values persist on ESP32 across generation steps.')
    print('LIMITATION: all eight weight sets and the full vocabulary head are still streamed each step.')
    print('STORAGE: PC streams weights for now. SD-card-only standalone inference is a future phase.')

if __name__=='__main__': main()
