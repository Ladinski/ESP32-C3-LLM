#!/usr/bin/env python3
"""Phase 14: execute layer zero of TinyStories-1M GPT-Neo on ESP32-C3.

Host selects pretrained tensor rows, sends weights and verifies outputs.
No model arithmetic is offloaded from the ESP32. NumPy is only a reference.
Requires a successful Phase 13 export (phase13_export/manifest.json).
"""
import argparse
import hashlib
import json
import math
import struct
import time
from pathlib import Path

import numpy as np

T, D, H, HD, F = 3, 64, 16, 4, 256
PREFIX = 'transformer.h.0.'


def fnv(payload):
    h = 2166136261
    for b in payload:
        h = ((h ^ b) * 16777619) & 0xffffffff
    return h


class Export:
    def __init__(self, root):
        self.root = Path(root)
        meta = json.loads((self.root / 'manifest.json').read_text(encoding='utf-8'))
        self.config = meta['source_config']
        self.items = {i['name']: i for i in meta['tensors']}
        if meta['source_repo'] != 'roneneldan/TinyStories-1M':
            raise RuntimeError('Unexpected pretrained checkpoint')
        if self.config['hidden_size'] != D or self.config['num_heads'] != H:
            raise RuntimeError('Model dimensions changed')
        if self.config.get('attention_layers', [None])[0] != 'global':
            raise RuntimeError('First layer must be global attention')
        self.activation = self.config.get('activation_function', 'gelu_new')
        if self.activation not in ('gelu_new', 'gelu'):
            raise RuntimeError('Unsupported GPT-Neo activation: ' + self.activation)
        self._cache = {}

    def item(self, name):
        if name not in self.items:
            raise RuntimeError('Required pretrained tensor missing: ' + name)
        return self.items[name]

    def raw(self, name):
        item = self.item(name)
        data = (self.root / item['file']).read_bytes()
        if hashlib.sha256(data).hexdigest() != item['sha256']:
            raise RuntimeError('SHA256 mismatch: ' + name)
        return data

    def fp(self, name, shape):
        item = self.item(name)
        if item['encoding'] != 'float16_le' or item['shape'] != list(shape):
            raise RuntimeError('Expected FP16 parameter ' + name + ': ' + str(item['shape']))
        return np.frombuffer(self.raw(name), dtype='<f2').astype(np.float32).reshape(shape)

    def matrix(self, name, shape):
        if name in self._cache:
            return self._cache[name]
        item = self.item(name)
        if item['encoding'] != 'int4_per_row' or item['shape'] != list(shape):
            raise RuntimeError('Expected INT4 matrix ' + name + ': ' + str(item['shape']))
        raw = np.frombuffer(self.raw(name), dtype=np.uint8).reshape(shape[0], -1)
        scales = np.fromfile(self.root / item['scale_file'], dtype='<f4')
        if scales.size != shape[0] or (scales <= 0).any() or not np.isfinite(scales).all():
            raise RuntimeError('Bad scales: ' + name)
        lo = (raw & 15).astype(np.int16)
        hi = (raw >> 4).astype(np.int16)
        lo[lo >= 8] -= 16
        hi[hi >= 8] -= 16
        q = np.empty((shape[0], raw.shape[1] * 2), dtype=np.int16)
        q[:, 0::2], q[:, 1::2] = lo, hi
        w = q[:, :shape[1]].astype(np.float32) * scales[:, None]
        self._cache[name] = w
        return w

    def row_packet(self, name, row):
        item = self.item(name)
        if item['encoding'] != 'int4_per_row':
            raise RuntimeError('Expected INT4: ' + name)
        rows, cols = item['shape']
        if row < 0 or row >= rows:
            raise RuntimeError('Row index out of range')
        rb = (cols + 1) // 2
        # File offsets are validated against SHA256 when matrix() loads each tensor.
        data = self.raw(name)
        scale = (self.root / item['scale_file']).read_bytes()
        return data[row * rb:(row + 1) * rb] + scale[row * 4:(row + 1) * 4]

    def packets_for_matrix(self, name, shape):
        self.matrix(name, shape)  # Validate shape and decode for independent reference.
        item = self.item(name)
        raw = self.raw(name)
        scales = (self.root / item['scale_file']).read_bytes()
        stride = (shape[1] + 1) // 2
        if len(scales) != shape[0] * 4:
            raise RuntimeError('Scale length mismatch: ' + name)
        for row in range(shape[0]):
            yield raw[row * stride:(row + 1) * stride] + scales[row * 4:(row + 1) * 4]


def layernorm(x, g, b, eps):
    mean = x.mean(axis=-1, keepdims=True, dtype=np.float32)
    var = np.mean((x - mean) ** 2, axis=-1, keepdims=True, dtype=np.float32)
    return (x - mean) / np.sqrt(var + eps) * g + b


def ref_block(e, ids, records):
    eps = float(e.config.get('layer_norm_epsilon', 1e-5))
    wte = e.matrix('transformer.wte.weight', (int(e.config['vocab_size']), D))
    wpe = e.matrix('transformer.wpe.weight', (int(e.config['max_position_embeddings']), D))
    x = wte[np.array(ids)] + wpe[np.arange(T)]
    ln1 = PREFIX + 'ln_1.'
    ln2 = PREFIX + 'ln_2.'
    attn = PREFIX + 'attn.attention.'
    mlp = PREFIX + 'mlp.'
    z = layernorm(x, e.fp(ln1+'weight', (D,)), e.fp(ln1+'bias', (D,)), eps)
    q = z @ e.matrix(attn+'q_proj.weight', (D,D)).T
    k = z @ e.matrix(attn+'k_proj.weight', (D,D)).T
    v = z @ e.matrix(attn+'v_proj.weight', (D,D)).T
    ctx = np.empty((T,D), dtype=np.float32)
    for t in range(T):
        for h in range(H):
            start = h * HD
            scores = np.array([np.dot(q[t, start:start+HD],k[s, start:start+HD]) / math.sqrt(HD)
                               for s in range(t+1)], dtype=np.float32)
            probs = np.exp(scores - scores.max()); probs /= probs.sum()
            ctx[t,start:start+HD] = probs @ v[:t+1,start:start+HD]
    out = ctx @ e.matrix(attn+'out_proj.weight', (D,D)).T + e.fp(attn+'out_proj.bias',(D,))
    x = x + out
    z = layernorm(x,e.fp(ln2+'weight',(D,)),e.fp(ln2+'bias',(D,)),eps)
    a = z @ e.matrix(mlp+'c_fc.weight',(F,D)).T + e.fp(mlp+'c_fc.bias',(F,))
    if e.activation == 'gelu_new':
        a = 0.5 * a * (1 + np.tanh(np.sqrt(2/np.pi) * (a+0.044715*a**3)))
    else:
        # exact GELU; no SciPy dependency
        a = 0.5*a*(1 + np.vectorize(math.erf, otypes=[np.float64])(a / math.sqrt(2)).astype(np.float32))
    y = a @ e.matrix(mlp+'c_proj.weight',(D,F)).T + e.fp(mlp+'c_proj.bias',(D,))
    return (x + y).astype(np.float32)


def payloads(e, ids):
    packets = []
    # Embedding lookup is transport-side selection; mathematical dequantization and addition happen on ESP32.
    for t, tid in enumerate(ids):
        packets.append(e.row_packet('transformer.wte.weight', tid))
        packets.append(e.row_packet('transformer.wpe.weight', t))
    def fp_packet(name, shape):
        packets.append(e.fp(name,shape).astype('<f4').tobytes())
    def matrix_packets(name, shape):
        packets.extend(e.packets_for_matrix(name,shape))
    fp_packet(PREFIX+'ln_1.weight',(D,))
    fp_packet(PREFIX+'ln_1.bias',(D,))
    for part in ('q_proj','k_proj','v_proj'):
        matrix_packets(PREFIX+'attn.attention.'+part+'.weight',(D,D))
    fp_packet(PREFIX+'attn.attention.out_proj.bias',(D,))
    matrix_packets(PREFIX+'attn.attention.out_proj.weight',(D,D))
    fp_packet(PREFIX+'ln_2.weight',(D,))
    fp_packet(PREFIX+'ln_2.bias',(D,))
    fp_packet(PREFIX+'mlp.c_fc.bias',(F,))
    matrix_packets(PREFIX+'mlp.c_fc.weight',(F,D))
    fp_packet(PREFIX+'mlp.c_proj.bias',(D,))
    matrix_packets(PREFIX+'mlp.c_proj.weight',(D,F))
    return packets


def expect(port, expected, timeout_s=10, context=''):
    """Wait for a firmware response, with an absolute deadline and useful context."""
    deadline = time.monotonic() + timeout_s
    latest = []
    while time.monotonic() < deadline:
        line = port.readline().decode('ascii', errors='replace').strip()
        if not line:
            continue
        if line.startswith('ESP32-C3 PHASE 14'):
            continue
        if line.startswith('STAGE '):
            print('ESP32:', line, flush=True)
            continue
        if line.startswith('ERR '):
            raise RuntimeError(f'ESP32 error during {context or expected}: {line}')
        if line.startswith(expected):
            return line
        latest.append(line)
        latest = latest[-4:]
        # Boot output, an unexpected ACK, or a crash report are never discarded.
        if line.startswith(('ESP-ROM:', 'rst:', 'Guru Meditation', 'E (', 'I (')):
            raise RuntimeError(f'ESP32 reset/log appeared during {context or expected}: {line}')
        raise RuntimeError(f'Expected {expected!r} during {context or expected}, received {line!r}')
    raise TimeoutError(
        f'No response {expected!r} within {timeout_s}s during {context or expected}. '
        f'Port={port.port}; previously seen: {latest}. '
        'ESP32 may have reset, stalled, or failed to send ACK.'
    )


def packet_description(index):
    """Human-readable 1-based packet sequence matching payloads()."""
    i = index
    if i <= 6:
        return f'input token {(i-1)//2}, {"word embedding" if i % 2 else "position embedding"}'
    i -= 6
    if i <= 2:
        return f'first layernorm {"gamma" if i == 1 else "beta"}'
    i -= 2
    for label, rows in (('query projection', 64), ('key projection', 64), ('value projection', 64),
                        ('attention output bias', 1), ('attention output projection', 64),
                        ('second layernorm gamma', 1), ('second layernorm beta', 1),
                        ('MLP input bias', 1), ('MLP input projection', 256),
                        ('MLP output bias', 1), ('MLP output projection', 64)):
        if i <= rows:
            return f'{label}, row {i}/{rows}' if rows > 1 else label
        i -= rows
    return 'unknown packet'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--export-dir', default='phase13_export')
    ap.add_argument('--port', default='COM3')
    ap.add_argument('--token-ids', nargs=3, type=int, default=[464, 2068, 7586], metavar='ID')
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args()
    e = Export(args.export_dir)
    ids = args.token_ids
    if any(t < 0 or t >= e.config['vocab_size'] for t in ids):
        raise RuntimeError('Token ID outside vocabulary')
    if float(e.config.get('layer_norm_epsilon', 1e-5)) != 1e-5:
        raise RuntimeError('Firmware LN epsilon differs from model: adjust before running')
    print('ESP32-C3 PHASE 14 | Real pretrained GPT-Neo layer 0')
    print('Checkpoint:', e.config['architectures'][0], '| token IDs:', ids, '| attention heads:', H)
    print('Activation:',e.activation, '| 3 token positions, pretrained INT4-per-row weights')
    ref = ref_block(e,ids,None)
    if ref.shape != (T,D) or not np.isfinite(ref).all():
        raise RuntimeError('PC reference contains invalid outputs')
    packets = payloads(e, ids)
    data = b''.join(packets)
    print(f'PC reference: finite {T}x{D} output | {len(packets)} packets | {len(data):,} bytes | FNV32 {fnv(data):08x}')
    if args.self_test:
        print('REFERENCE PASS: all pretrained tensors decoded and one full layer executed on PC')
        print('Reference final token, first 8 features:',ref[-1,:8].tolist())
        return
    import serial
    with serial.Serial(args.port,115200,timeout=0.3,write_timeout=10) as port:
        port.reset_input_buffer()
        port.write(b'HELLO\n'); print(expect(port,'READY PHASE14'))
        port.write(b'RUN EXACT\n' if e.activation == 'gelu' else b'RUN NEW\n'); print(expect(port,'GO PHASE14'))
        start = time.perf_counter()
        for i,payload in enumerate(packets, 1):
            port.write(payload)
            expect(port,'ACK', timeout_s=10, context=f'record {i}/{len(packets)} ({packet_description(i)}, {len(payload)} bytes)')
            if i % 128 == 0 or (i >= 480 and i % 16 == 0):
                print(f'  streamed {i}/{len(packets)} records | {packet_description(i)}', flush=True)
        actual = np.empty((T,D),dtype=np.float32)
        for t in range(T):
            for chunk in range(4):
                line=expect(port,f'OUT {t} {chunk} ',timeout_s=30,
                            context=f'output row {t}, chunk {chunk}')
                parts=line.split()
                if len(parts) != 19 or int(parts[1]) != t or int(parts[2]) != chunk:
                    raise RuntimeError('Bad device output chunk: ' + line)
                actual[t,chunk*16:(chunk+1)*16]=np.array([float(v) for v in parts[3:]],dtype=np.float32)
        done = expect(port,'DONE ',timeout_s=30,context='final DONE')
        elapsed = time.perf_counter()-start
    parts=done.split()
    if len(parts)!=7:
        raise RuntimeError('Malformed device DONE: '+done)
    count,device_hash=int(parts[1]),int(parts[2],16)
    if count != len(data) or device_hash != fnv(data):
        raise RuntimeError('Transfer byte count or hash mismatch: '+done)
    error = float(np.max(np.abs(actual-ref)))
    print('Last-token ESP32, first 8 features:',actual[-1,:8].tolist())
    print('Last-token Python, first 8 features:',ref[-1,:8].tolist())
    print(f'Max absolute difference vs exported-INT4 Python: {error:.7f}')
    print(f'Host elapsed: {elapsed:.3f}s | ESP32: {int(parts[6])/1e6:.3f}s')
    print(f'ESP32 linears: {int(parts[3])/1000:.2f}ms | other math: {int(parts[4])/1000:.2f}ms | receive/wait: {int(parts[5])/1000:.2f}ms')
    if error > 0.02:
        raise RuntimeError('Numerical mismatch exceeds 0.02 absolute tolerance. Do not proceed to full model.')
    print('PHASE 14 PASS: one complete real pretrained GPT-Neo block computed on ESP32-C3')
    print('NOTE: layer 0 only; no language generation or full pretrained model inference yet.')


if __name__=='__main__':
    main()
