"""Phase 10: autoregressive INT4 decoder with persistent KV cache on ESP32-C3.

Self-test: uv run --python 3.12 --with numpy python token_generator.py --self-test
Board:     uv run --python 3.12 --with numpy --with pyserial python token_generator.py
"""
import argparse
import time
import numpy as np

VOCAB, DIM, FF, MAX_TOKENS = 16, 16, 32, 12
SCALE, EPS = 16.0, 1e-5
SEED = 20261113
FNV_OFFSET, FNV_PRIME = 2166136261, 16777619
CHARS = "\n !023adehloprsw"


def fnv32(data):
    h = FNV_OFFSET
    for byte in data:
        h = ((h ^ byte) * FNV_PRIME) & 0xFFFFFFFF
    return h


def pack4(w):
    assert w.dtype == np.int8 and w.shape[-1] % 2 == 0
    return ((w[..., ::2].astype(np.uint8) & 15) |
            ((w[..., 1::2].astype(np.uint8) & 15) << 4)).astype(np.uint8).tobytes()


def unpack_model(payload):
    if len(payload) != 1408:
        raise ValueError(f"Expected exactly 1408 bytes, got {len(payload)}")
    emb = np.frombuffer(payload[:256], dtype=np.int8).reshape(VOCAB, DIM).copy()
    weights, offset = [], 256
    for rows, cols in [(DIM, DIM)] * 4 + [(FF, DIM), (DIM, FF), (VOCAB, DIM)]:
        length = rows * cols // 2
        packed = np.frombuffer(payload[offset:offset + length], dtype=np.uint8)
        unpacked = np.empty(rows * cols, dtype=np.int8)
        unpacked[::2] = ((packed & 15).astype(np.int8) ^ 8) - 8
        unpacked[1::2] = (((packed >> 4).astype(np.int8)) ^ 8) - 8
        weights.append(unpacked.reshape(rows, cols))
        offset += length
    assert offset == len(payload)
    return emb, weights


def norm(x):
    return (x - np.mean(x, axis=-1, keepdims=True)) / np.sqrt(
        np.mean((x - np.mean(x, axis=-1, keepdims=True)) ** 2,
                axis=-1, keepdims=True) + EPS)


def linear(x, w):
    return (x @ w.astype(np.float64).T) / SCALE


def embedding(emb, token, position):
    x = emb[token].astype(np.float64) / 8.0
    x[np.arange(DIM) % 4 == position % 4] += 0.125
    return x


def cached_decode(emb, ws, seed, count):
    k_cache, v_cache = [], []
    selected, used, scores = [], [], []
    token = seed
    wq, wk, wv, wo, w1, w2, head = ws
    for pos in range(count):
        used.append(token)
        x = embedding(emb, token, pos)
        n = norm(x)
        q, k, v = (linear(n, w) for w in (wq, wk, wv))
        k_cache.append(k)
        v_cache.append(v)
        logit = np.asarray(k_cache) @ q / np.sqrt(float(DIM))
        p = np.exp(logit - np.max(logit))
        p /= p.sum()
        context = p @ np.asarray(v_cache)
        residual = x + linear(context, wo)
        hidden = np.maximum(linear(norm(residual), w1), 0)
        final = norm(residual + linear(hidden, w2))
        logits = linear(final, head)
        predicted = int(np.argmax(logits))
        scores.append(logits)
        selected.append(predicted)
        token = predicted
    return used, selected, np.asarray(scores)


def full_prefix_decode(emb, ws, used):
    # Independently recompute every Q/K/V from the entire prefix at each step.
    wq, wk, wv, wo, w1, w2, head = ws
    outputs = []
    for end in range(1, len(used) + 1):
        x = np.stack([embedding(emb, used[i], i) for i in range(end)])
        n = norm(x)
        q, k, v = (linear(n, w) for w in (wq, wk, wv))
        scores = (q[-1] @ k.T) / np.sqrt(float(DIM))
        weights = np.exp(scores - scores.max())
        weights /= weights.sum()
        context = weights @ v
        residual = x[-1] + linear(context, wo)
        hidden = np.maximum(linear(norm(residual), w1), 0)
        outputs.append(linear(norm(residual + linear(hidden, w2)), head))
    return np.asarray(outputs)


def expect(port, prefix, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        line = port.readline().decode('ascii', errors='replace').strip()
        if line.startswith('ERR '):
            raise RuntimeError('ESP32: ' + line)
        if line.startswith(prefix):
            return line
    raise TimeoutError(f'No {prefix} reply. Close ESP-IDF monitor and verify Phase 11 firmware.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', default='COM3')
    parser.add_argument('--seed', type=int, default=CHARS.index("h"))
    parser.add_argument('--steps', type=int, default=12)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if not 0 <= args.seed < VOCAB or not 1 <= args.steps <= MAX_TOKENS:
        parser.error(f'--seed must be 0..{VOCAB - 1}; --steps must be 1..{MAX_TOKENS}')
    from pathlib import Path
    path = Path(__file__).resolve().parent / "phase11_model.bin"
    if not path.exists():
        raise SystemExit("Model missing. First run: python train_language_model.py")
    payload = path.read_bytes()
    emb, ws = unpack_model(payload)
    used, predicted, expected = cached_decode(emb, ws, args.seed, args.steps)
    full_reference = full_prefix_decode(emb, ws, used)
    crosscheck = float(np.max(np.abs(expected - full_reference)))
    assert crosscheck < 1e-10, f'Cache and full-prefix reference differ: {crosscheck}'
    print('ESP32-C3 Phase 11 | TRAINED character transformer + KV cache')
    print('16-character vocabulary; 16 channels; INT4 weights; 12-token KV cache')
    print('TRAINED tiny character transformer; memorized training string, not conversational AI')
    print(f'Model: {len(payload)} bytes | FNV32: {fnv32(payload):08x}')
    print(f'Python cache-vs-full-prefix max error: {crosscheck:.3g}')
    if args.self_test:
        for pos, (tok, chosen) in enumerate(zip(used, predicted)):
            print(f'[{pos}] {repr(CHARS[tok])} -> {repr(CHARS[chosen])}')
        print('REFERENCE PASS: autoregressive cache exactly matches full-prefix computation')
        return
    import serial
    with serial.Serial(args.port, 115200, timeout=0.25, write_timeout=10) as esp:
        time.sleep(.3)
        esp.reset_input_buffer()
        esp.write(b'HELLO\n')
        print(expect(esp, 'READY PHASE11'))
        esp.write(b'LOAD\n')
        response = expect(esp, 'GO ')
        assert response == f'GO {len(payload)}', f'Wrong firmware/model payload: {response}'
        for offset in range(0, len(payload), 256):
            chunk = payload[offset:offset + 256]
            esp.write(chunk)
            assert expect(esp, 'ACK ') == f'ACK {offset + len(chunk)}'
        loaded = expect(esp, 'LOADED ').split()
        assert len(loaded) == 3 and int(loaded[1]) == len(payload)
        assert int(loaded[2], 16) == fnv32(payload), 'Model byte hash mismatch'
        print('PASS: model transferred and hash verified')
        esp.write(f'GENERATE {args.seed} {args.steps}\n'.encode())
        started = time.perf_counter()
        actual = []
        for pos in range(args.steps):
            line = expect(esp, 'SCORES ').split()
            assert len(line) == 2 + VOCAB and int(line[1]) == pos, line
            scores = np.array([float(s) for s in line[2:]])
            actual.append(scores)
            step = expect(esp, 'STEP ').split()
            assert len(step) == 4 and int(step[1]) == pos
            _, _, input_token, output_token = step
            actual_in, actual_out = int(input_token), int(output_token)
            assert actual_in == used[pos], (pos, actual_in, used[pos])
            assert actual_out == predicted[pos], (pos, actual_out, predicted[pos])
            print(f'[{pos + 1:2}/{args.steps}] {repr(CHARS[actual_in]):>7} -> {repr(CHARS[actual_out]):<7} PASS')
        finished = expect(esp, 'DONE ').split()
        total_ms = (time.perf_counter() - started) * 1000
    assert len(finished) == 6, finished
    _, steps, remote_hash, linear_us, attn_us, device_us = finished
    assert int(steps) == args.steps
    assert int(remote_hash, 16) == fnv32(payload)
    error = float(np.max(np.abs(np.asarray(actual) - expected)))
    print(f'Max ESP32 vs Python logit difference: {error:.6f}')
    print(f'USB + decoding: {total_ms:.3f} ms | ESP32 elapsed: {int(device_us)/1000:.3f} ms')
    print(f'ESP32 INT4 linears: {int(linear_us)/1000:.3f} ms | attention: {int(attn_us)/1000:.3f} ms')
    if error > 0.003:
        raise AssertionError('ESP32 logits differ from Python')
    print('Generated text:', repr(CHARS[args.seed] + ''.join(CHARS[t] for t in predicted)))
    print('PASS: all generated token IDs and logits match Python; persistent KV cache exercised')
    print('PHASE 11 TRAINED CHARACTER GENERATION COMPLETE')


if __name__ == '__main__':
    main()
