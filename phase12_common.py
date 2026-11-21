"""Phase 12: extended vocabulary INT4 decoder with persistent KV cache on ESP32-C3.

Self-test: uv run --python 3.12 --with numpy python token_generator.py --self-test
Board:     uv run --python 3.12 --with numpy --with pyserial python token_generator.py
"""
import argparse
import time
import numpy as np

VOCAB, DIM, FF, MAX_TOKENS = 32, 16, 32, 24
SCALE, EPS = 16.0, 1e-5
SEED = 20261113
FNV_OFFSET, FNV_PRIME = 2166136261, 16777619
CHARS = "\n !.0123456789abcdefghilmnoprstw"


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
    if len(payload) != 1792:
        raise ValueError(f"Expected exactly 1792 bytes, got {len(payload)}")
    emb = np.frombuffer(payload[:512], dtype=np.int8).reshape(VOCAB, DIM).copy()
    weights, offset = [], 512
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



def prompt_decode(emb, ws, prompt, steps):
    """Reference: prefill prompt, then greedily predict; one cache across steps."""
    assert len(prompt) >= 1 and len(prompt) + steps - 1 <= MAX_TOKENS
    wq, wk, wv, wo, w1, w2, head = ws
    keys, values = [], []
    generated, traces, seen = [], [], []
    inputs = list(prompt)
    for pos in range(len(prompt) + steps - 1):
        token = inputs[pos]
        seen.append(token)
        x = embedding(emb, token, pos)
        z = norm(x)
        q, k, v = (linear(z, w) for w in (wq, wk, wv))
        keys.append(k); values.append(v)
        raw = np.asarray(keys) @ q / np.sqrt(float(DIM))
        p = np.exp(raw - np.max(raw)); p /= p.sum()
        context = p @ np.asarray(values)
        residual = x + linear(context, wo)
        hidden = np.maximum(linear(norm(residual), w1), 0)
        scores = linear(norm(residual + linear(hidden, w2)), head)
        if pos >= len(prompt) - 1:
            nxt = int(np.argmax(scores))
            generated.append(nxt)
            traces.append(scores)
            inputs.append(nxt)
    return seen, generated, np.asarray(traces)

TRAIN = [
    'hi esp32!\n', 'hello esp32!\n', 'esp32 is fast.\n',
    'esp32 is small.\n', 'hello there!\n', 'hi there!\n',
    'esp32 can learn.\n',
]
HELDOUT = ['hi esp32.\n', 'hello esp32.\n', 'esp32 is real.\n']
assert len(CHARS) == VOCAB and all(set(s) <= set(CHARS) and len(s) <= MAX_TOKENS + 1 for s in TRAIN + HELDOUT)
