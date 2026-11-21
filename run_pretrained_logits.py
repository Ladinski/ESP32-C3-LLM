#!/usr/bin/env python3
"""Phase 16: eight GPT-Neo blocks, final normalization, full-vocabulary streamed top-5 on ESP32-C3.

PC stores, selects and transmits weights; PC computes reference strictly for verification.
ESP32 performs all neural operations. Three fixed token positions, no generation yet.
"""
import argparse
import math
import time
import numpy as np
from phase15_core import Export, D, H, HD, F, T, fnv, layernorm, expect

LAYERS = 8

def block(e, x, index):
    p = f'transformer.h.{index}.'
    attn = p + 'attn.attention.'
    mlp = p + 'mlp.'
    eps = float(e.config.get('layer_norm_epsilon', 1e-5))
    z = layernorm(x, e.fp(p+'ln_1.weight',(D,)), e.fp(p+'ln_1.bias',(D,)), eps)
    q = z @ e.matrix(attn+'q_proj.weight',(D,D)).T
    k = z @ e.matrix(attn+'k_proj.weight',(D,D)).T
    v = z @ e.matrix(attn+'v_proj.weight',(D,D)).T
    ctx = np.empty((T,D), dtype=np.float32)
    local_window = int(e.config.get('window_size', 256))
    is_local = e.config['attention_layers'][index] == 'local'
    for t in range(T):
        first = max(0,t-local_window+1) if is_local else 0
        for h in range(H):
            start = h*HD
            scores = np.array([np.dot(q[t,start:start+HD], k[s,start:start+HD]) / math.sqrt(HD)
                               for s in range(first,t+1)],dtype=np.float32)
            probs = np.exp(scores-scores.max()); probs /= probs.sum()
            ctx[t,start:start+HD] = probs @ v[first:t+1,start:start+HD]
    x = x + ctx @ e.matrix(attn+'out_proj.weight',(D,D)).T + e.fp(attn+'out_proj.bias',(D,))
    z = layernorm(x,e.fp(p+'ln_2.weight',(D,)),e.fp(p+'ln_2.bias',(D,)),eps)
    a = z @ e.matrix(mlp+'c_fc.weight',(F,D)).T + e.fp(mlp+'c_fc.bias',(F,))
    if e.activation == 'gelu_new':
        a = 0.5*a*(1+np.tanh(np.sqrt(2/np.pi)*(a+0.044715*a**3)))
    else:
        a = 0.5*a*(1+np.vectorize(math.erf,otypes=[np.float64])(a/math.sqrt(2)).astype(np.float32))
    y = a @ e.matrix(mlp+'c_proj.weight',(D,F)).T + e.fp(mlp+'c_proj.bias',(D,))
    return np.asarray(x+y,dtype=np.float32)


def embedding_packets(e,ids):
    packets=[]
    for t,tid in enumerate(ids):
        packets.append(e.row_packet('transformer.wte.weight',tid))
        packets.append(e.row_packet('transformer.wpe.weight',t))
    return packets


def layer_packets(e,index):
    p=f'transformer.h.{index}.'
    attn=p+'attn.attention.'
    mlp=p+'mlp.'
    packets=[]
    def fp(name,shape):
        packets.append(e.fp(name,shape).astype('<f4').tobytes())
    def matrix(name,shape):
        packets.extend(e.packets_for_matrix(name,shape))
    fp(p+'ln_1.weight',(D,)); fp(p+'ln_1.bias',(D,))
    for part in ('q_proj','k_proj','v_proj'):
        matrix(attn+part+'.weight',(D,D))
    fp(attn+'out_proj.bias',(D,))
    matrix(attn+'out_proj.weight',(D,D))
    fp(p+'ln_2.weight',(D,)); fp(p+'ln_2.bias',(D,))
    fp(mlp+'c_fc.bias',(F,)); matrix(mlp+'c_fc.weight',(F,D))
    fp(mlp+'c_proj.bias',(D,)); matrix(mlp+'c_proj.weight',(D,F))
    assert len(packets)==583
    return packets


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--export-dir',default='phase13_export')
    ap.add_argument('--port',default='COM3')
    ap.add_argument('--token-ids',type=int,nargs=3,default=[464,2068,7586])
    ap.add_argument('--self-test',action='store_true')
    args=ap.parse_args()
    e=Export(args.export_dir)
    cfg=e.config
    if int(cfg['num_layers'])!=LAYERS:
        raise RuntimeError('Expected exactly 8 pretrained layers')
    if list(cfg['attention_layers'])!=['global','local']*4:
        raise RuntimeError('Unexpected GPT-Neo attention pattern')
    if int(cfg.get('window_size',256))<T:
        raise RuntimeError('Phase 15 uses 3 tokens: local window must be >=3')
    if float(cfg.get('layer_norm_epsilon',1e-5))!=1e-5:
        raise RuntimeError('LayerNorm epsilon mismatch')
    if any(i<0 or i>=int(cfg['vocab_size']) for i in args.token_ids):
        raise RuntimeError('Invalid input token ID')
    print('ESP32-C3 PHASE 16 | Eight pretrained blocks + full-vocabulary top-5')
    print('Tokens:',args.token_ids,'| attention:',','.join(cfg['attention_layers']))
    print('Activation:',e.activation,'| PC reference used only for comparison')
    wte=e.matrix('transformer.wte.weight',(int(cfg['vocab_size']),D))
    wpe=e.matrix('transformer.wpe.weight',(int(cfg['max_position_embeddings']),D))
    x=(wte[np.array(args.token_ids)]+wpe[np.arange(T)]).astype(np.float32)
    references=[]
    packet_sets=[]
    for i in range(LAYERS):
        x=block(e,x,i)
        if x.shape!=(T,D) or not np.isfinite(x).all():
            raise RuntimeError(f'Nonfinite Python reference output for layer {i}')
        references.append(x.copy())
        packet_sets.append(layer_packets(e,i))
        print(f'  Python layer {i} ready | last token first feature {x[-1,0]:.7g} | {len(packet_sets[-1])} records')
    # TinyStories GPT-Neo uses tied input/output embeddings. The exported state
    # must not contain a separate independently trained lm_head tensor.
    vocab=int(cfg['vocab_size'])
    if vocab!=50257:
        raise RuntimeError(f'Firmware expects vocabulary 50257, got {vocab}')
    if 'lm_head.bias' in e.items:
        raise RuntimeError('Unexpected output vocabulary bias; firmware assumes none')
    if 'lm_head.weight' in e.items:
        raise RuntimeError('Checkpoint contains a separate lm_head.weight: verify tying before scoring')
    ln_final_w=e.fp('transformer.ln_f.weight',(D,))
    ln_final_b=e.fp('transformer.ln_f.bias',(D,))
    normalized=layernorm(references[-1],ln_final_w,ln_final_b,float(cfg.get('layer_norm_epsilon',1e-5)))
    # Independent full-vocabulary NumPy baseline. The ESP32 computes every score itself.
    embedding=e.matrix('transformer.wte.weight',(vocab,D))
    logits=embedding @ normalized[-1]
    if not np.isfinite(logits).all(): raise RuntimeError('Nonfinite Python reference logits')
    top_ids=sorted(range(vocab),key=lambda i:(-float(logits[i]),i))[:5]
    print('PC reference top-5 token IDs:',top_ids)
    print('PC reference top-5 logits:',[float(logits[i]) for i in top_ids])
    # Cache the verified quantized weights once; row_packet() hashes the entire
    # embedding on each call, so it must not be used 50,257 times.
    wte_item=e.item('transformer.wte.weight')
    raw=e.raw('transformer.wte.weight')
    scales=(e.root/wte_item['scale_file']).read_bytes()
    if len(raw)!=vocab*(D//2) or len(scales)!=vocab*4:
        raise RuntimeError('Vocabulary embedding payload size mismatch')
    vocab_payload=[raw[i*32:(i+1)*32]+scales[i*4:(i+1)*4] for i in range(vocab)]
    head_packets=[ln_final_w.astype('<f4').tobytes(),ln_final_b.astype('<f4').tobytes()]+vocab_payload
    first=embedding_packets(e,args.token_ids)
    combined=first+[p for group in packet_sets for p in group]+head_packets
    total_bytes=sum(map(len,combined))
    checksum=fnv(b''.join(combined))
    print(f'Payload: {len(combined):,} records | {total_bytes:,} bytes | FNV32 {checksum:08x}')
    if args.self_test:
        print('PHASE 16 REFERENCE PASS: eight quantized blocks + full 50,257-token logits')
        print('Last-token first 8 features:',references[-1][-1,:8].tolist())
        return
    import serial
    with serial.Serial(args.port,115200,timeout=0.3,write_timeout=10) as port:
        port.reset_input_buffer()
        port.write(b'HELLO\n'); print(expect(port,'READY PHASE16'))
        port.write(b'RUN EXACT\n' if e.activation=='gelu' else b'RUN NEW\n')
        print(expect(port,'GO PHASE16'))
        started=time.perf_counter()
        sent=0
        def stream(packets,lab):
            nonlocal sent
            for i,payload in enumerate(packets,1):
                port.write(payload)
                expect(port,'ACK',timeout_s=10,context=f'{lab}, packet {i}/{len(packets)} ({len(payload)} bytes)')
                sent+=1
                if i%128==0:
                    print(f'  {lab}: {i}/{len(packets)} packets',flush=True)
        stream(first,'embeddings')
        for layer in range(LAYERS):
            if layer:
                port.write(f'NEXT {layer}\n'.encode('ascii'))
                print(expect(port,f'GO LAYER {layer}',context=f'layer {layer} start'))
            stream(packet_sets[layer],f'layer {layer}')
            actual=np.empty((T,D),dtype=np.float32)
            for t in range(T):
                for chunk in range(4):
                    line=expect(port,f'OUT {layer} {t} {chunk} ',timeout_s=30,
                                context=f'layer {layer}, token {t}, chunk {chunk}')
                    fields=line.split()
                    if len(fields)!=20 or fields[1:4]!=[str(layer),str(t),str(chunk)]:
                        raise RuntimeError('Malformed output: '+line)
                    actual[t,chunk*16:(chunk+1)*16]=np.asarray([float(z) for z in fields[4:]],dtype=np.float32)
            print(expect(port,f'LAYER_DONE {layer}',context=f'layer {layer} completion'))
            err=float(np.max(np.abs(actual-references[layer])))
            print(f'  LAYER {layer} max absolute error: {err:.8g}',flush=True)
            if not np.isfinite(actual).all() or err>0.02:
                raise RuntimeError(f'FAILED layer {layer}: ESP32 differs from quantized reference')
        print(expect(port,'LAYERS_DONE',timeout_s=30,context='all layers complete'))
        port.write(b'HEAD 50257\n')
        print(expect(port,'GO HEAD',timeout_s=30,context='vocabulary-scoring start'))
        print('Streaming final LayerNorm and 50,257 vocabulary rows...')
        for i,payload in enumerate(head_packets,1):
            port.write(payload)
            expect(port,'ACK',timeout_s=10,context=f'head packet {i}/{len(head_packets)}')
            sent+=1
            if i>=3 and ((i-2)%4096==0 or i==len(head_packets)):
                print(f'  scored {i-2:,}/{vocab:,} vocabulary tokens',flush=True)
        device_top=[]
        for pos in range(5):
            line=expect(port,f'TOP {pos} ',timeout_s=30,context=f'top-{pos+1}')
            parts=line.split()
            if len(parts)!=4 or int(parts[1])!=pos:
                raise RuntimeError('Malformed TOP: '+line)
            device_top.append((int(parts[2]),float(parts[3])))
        done=expect(port,'DONE ',timeout_s=30,context='final vocabulary scoring')
        elapsed=time.perf_counter()-started
    fields=done.split()
    if len(fields)!=7: raise RuntimeError('Malformed DONE: '+done)
    if int(fields[1])!=total_bytes or int(fields[2],16)!=checksum:
        raise RuntimeError('Full transport byte-count/hash mismatch: '+done)
    print('All 8 layers: PASS | full byte count and FNV hash: PASS')
    print('ESP32 top-5:',device_top)
    print('Python top-5:',[(i,float(logits[i])) for i in top_ids])
    if [i for i,_ in device_top] != top_ids:
        raise RuntimeError('Vocabulary top-5 token IDs differ from Python')
    max_logit_err=max(abs(score-float(logits[idx])) for idx,score in device_top)
    print(f'Top-5 logit maximum absolute difference: {max_logit_err:.8g}')
    if max_logit_err>0.02:
        raise RuntimeError('Vocabulary logit error exceeds 0.02')
    print('Last-token ESP32 first 8:',actual[-1,:8].tolist())
    print('Last-token Python first 8:',references[-1][-1,:8].tolist())
    print(f'Host elapsed: {elapsed:.3f}s | ESP32 elapsed: {int(fields[6])/1e6:.3f}s')
    print(f'Linear compute: {int(fields[3])/1e3:.2f}ms | other math: {int(fields[4])/1e3:.2f}ms | receive/wait: {int(fields[5])/1e3:.2f}ms')
    print('PHASE 16 PASS: all eight pretrained blocks + final LN + all vocabulary logits computed on ESP32-C3')
    print(f'Greedy next token ID: {device_top[0][0]} (not yet decoded or autoregressively generated)')
    print('NOTE: fixed three-token prompt only; no iterative autoregressive generation yet.')

if __name__=='__main__':
    main()
