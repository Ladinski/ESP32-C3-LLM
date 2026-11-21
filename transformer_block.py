"""Phase 9: One pre-norm decoder transformer block on ESP32-C3.

Run: uv run --python 3.12 --with numpy --with pyserial python transformer_block.py
Reference only: uv run --python 3.12 --with numpy python transformer_block.py --self-test
"""
import argparse
import time
import numpy as np

TOKENS, DIM, FF = 4, 16, 32
SEED = 20261113
SCALE = 16.0
EPSILON = 1e-5
FNV_OFFSET, FNV_PRIME = 2166136261, 16777619


def fnv32(data):
    h = FNV_OFFSET
    for byte in data:
        h = ((h ^ byte) * FNV_PRIME) & 0xFFFFFFFF
    return h


def pack_int4(matrix):
    assert matrix.dtype == np.int8 and matrix.shape[1] % 2 == 0
    assert np.all((matrix >= -8) & (matrix <= 7))
    lo = matrix[:, 0::2].astype(np.uint8) & 0x0F
    hi = matrix[:, 1::2].astype(np.uint8) & 0x0F
    return (lo | (hi << 4)).astype(np.uint8).tobytes()


def create_payload():
    rng = np.random.default_rng(SEED)
    inputs = rng.integers(-16, 17, size=(TOKENS, DIM), dtype=np.int8)
    shapes = [(DIM, DIM)] * 4 + [(FF, DIM), (DIM, FF)]
    weights = [rng.integers(-7, 8, size=shape, dtype=np.int8) for shape in shapes]
    chunks = [inputs.tobytes()] + [pack_int4(w) for w in weights]
    return inputs, weights, chunks


def layernorm(x):
    mean = x.mean(axis=-1, keepdims=True)
    var = np.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + EPSILON)


def linear(x, w):
    return (x @ w.astype(np.float64).T) / SCALE


def reference(inputs, weights):
    x = inputs.astype(np.float64) / 8.0
    norm = layernorm(x)
    q, k, v = (linear(norm, weights[i]) for i in range(3))
    attn = np.empty((TOKENS, DIM), dtype=np.float64)
    probs = np.zeros((TOKENS, TOKENS), dtype=np.float64)
    for t in range(TOKENS):
        logits = (q[t] @ k[:t+1].T) / np.sqrt(float(DIM))
        logits = np.exp(logits - logits.max())
        probs[t, :t+1] = logits / logits.sum()
        attn[t] = probs[t, :t+1] @ v[:t+1]
    post_attn = x + linear(attn, weights[3])
    hidden = np.maximum(linear(layernorm(post_attn), weights[4]), 0)
    return post_attn + linear(hidden, weights[5]), probs


def expect(port, prefix, timeout=10):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        b = port.readline()
        if not b:
            continue
        line = b.decode('ascii', errors='replace').strip()
        if line.startswith('ERR '):
            raise RuntimeError('ESP32: ' + line)
        if line.startswith(prefix):
            return line
    raise TimeoutError(f'No {prefix} response. Close serial monitors and verify Phase 9 firmware.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', default='COM3')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    inputs, weights, chunks = create_payload()
    expected, probs = reference(inputs, weights)
    payload = b''.join(chunks)
    print('ESP32-C3 Phase 9 | Pre-norm decoder transformer block')
    print('Architecture: LN -> QKV -> causal attention -> output projection -> residual -> LN -> FFN -> residual')
    print('4 tokens, 16 channels, 32 FFN hidden; streamed packed INT4 matrices')
    print(f'Payload: {len(payload)} bytes | FNV32: {fnv32(payload):08x}')
    if args.self_test:
        assert all(np.isclose(row[:i+1].sum(), 1.0) for i, row in enumerate(probs))
        assert np.all(np.isfinite(expected))
        print('REFERENCE PASS: stable softmax, causal mask, finite block output')
        print('Python output row 3:', np.round(expected[3], 5).tolist())
        return

    import serial
    with serial.Serial(args.port, 115200, timeout=0.25, write_timeout=10) as esp:
        time.sleep(0.3)
        esp.reset_input_buffer()
        esp.write(b'HELLO\n')
        print(expect(esp, 'READY PHASE9'))
        esp.write(b'RUN\n')
        go = expect(esp, 'GO ')
        assert go == f'GO {len(payload)}', f'Wrong device payload size: {go}'
        start = time.perf_counter()
        transferred = 0
        for chunk in chunks:
            esp.write(chunk)
            transferred += len(chunk)
            ack = expect(esp, 'ACK ')
            assert ack == f'ACK {transferred}', f'Expected ACK {transferred}, received {ack}'
        rows = []
        for t in range(TOKENS):
            fields = expect(esp, 'ROW ').split()
            assert len(fields) == DIM + 2 and int(fields[1]) == t, f'Bad output row: {fields}'
            rows.append([float(v) for v in fields[2:]])
        parts = expect(esp, 'DONE ').split()
        total_ms = (time.perf_counter() - start) * 1000
    assert len(parts) == 7, f'Unexpected DONE: {parts}'
    _, byte_count, remote_hash, linear_us, other_us, receive_us, device_us = parts
    assert int(byte_count) == len(payload), 'Byte count mismatch'
    assert int(remote_hash, 16) == fnv32(payload), 'FNV32 mismatch'
    observed = np.asarray(rows)
    maximum_error = float(np.max(np.abs(observed - expected)))
    print('Transformer output row 3:', np.round(observed[3], 5).tolist())
    print(f'Max difference vs Python: {maximum_error:.6f}')
    print(f'End-to-end: {total_ms:.3f} ms | ESP32 elapsed: {int(device_us)/1000:.3f} ms')
    print(f'ESP32 INT4 linears: {int(linear_us)/1000:.3f} ms')
    print(f'ESP32 normalization/attention/residual/activation: {int(other_us)/1000:.3f} ms')
    print(f'ESP32 USB receive/wait: {int(receive_us)/1000:.3f} ms')
    if maximum_error > 0.003:
        raise AssertionError('Transformer outputs differ from reference')
    print('PASS: transfer hash matches and all 64 transformer outputs match Python reference')
    print('PHASE 9 FULL TRANSFORMER BLOCK COMPLETE')


if __name__ == '__main__':
    main()
