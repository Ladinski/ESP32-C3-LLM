"""Phase 8: stream INT4 Q/K/V weights; ESP32 runs causal self-attention.

Run: uv run --python 3.12 --with numpy --with pyserial python attention_stream.py
Offline reference: uv run --python 3.12 --with numpy python attention_stream.py --self-test
"""
import argparse
import time
import numpy as np

TOKENS = 4
DIM = 16
FNV_OFFSET = 2166136261
FNV_PRIME = 16777619


def fnv32(data):
    h = FNV_OFFSET
    for byte in data:
        h = ((h ^ byte) * FNV_PRIME) & 0xFFFFFFFF
    return h


def pack_int4(matrix):
    assert matrix.dtype == np.int8 and matrix.shape == (DIM, DIM)
    low = matrix[:, 0::2].astype(np.uint8) & 15
    high = matrix[:, 1::2].astype(np.uint8) & 15
    return (low | (high << 4)).astype(np.uint8).tobytes()


def test_data():
    rng = np.random.default_rng(20261113)
    x = rng.integers(-8, 9, size=(TOKENS, DIM), dtype=np.int8)
    # Rows of each matrix represent output channels.
    matrices = [rng.integers(-7, 8, size=(DIM, DIM), dtype=np.int8) for _ in range(3)]
    chunks = [x.astype(np.int8).tobytes()] + [pack_int4(w) for w in matrices]
    return x, matrices, chunks


def reference_attention(x, weights):
    xi = x.astype(np.int64)
    q, k, v = [(xi @ w.astype(np.int64).T) for w in weights]
    results = np.zeros((TOKENS, DIM), dtype=np.float64)
    probabilities = np.zeros((TOKENS, TOKENS), dtype=np.float64)
    for t in range(TOKENS):
        logits = (q[t] @ k[: t + 1].T).astype(np.float64) / (np.sqrt(DIM) * 16384.0)
        exponents = np.exp(logits - logits.max())
        probabilities[t, : t + 1] = exponents / exponents.sum()
        results[t] = probabilities[t, : t + 1] @ v[: t + 1]
    return results, probabilities


def expect(port, prefix, timeout=10.0):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        line_bytes = port.readline()
        if not line_bytes:
            continue
        line = line_bytes.decode('ascii', errors='replace').strip()
        if line.startswith('ERR '):
            raise RuntimeError(f'ESP32 reported {line}')
        if line.startswith(prefix):
            return line
    raise TimeoutError(f'No {prefix!r} response from ESP32. Check Phase 8 firmware and COM port.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', default='COM3')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()

    inputs, weights, chunks = test_data()
    reference, probs = reference_attention(inputs, weights)
    print('ESP32-C3 Phase 8 | Four-token causal self-attention')
    print('Q/K/V: streamed INT4 matrices; ESP32: integer projections + scaled softmax attention')
    print('Causal mask: token t can only attend to positions <= t')
    print('Payload: %d bytes | Host FNV32: %08x' % (sum(map(len, chunks)), fnv32(b''.join(chunks))))
    if args.self_test:
        print('OFFLINE REFERENCE PASS; each attention row sums to 1')
        print('Attention probabilities:\n', np.round(probs, 4))
        print('Python output row 3:', np.round(reference[3], 4).tolist())
        return

    import serial
    with serial.Serial(args.port, 115200, timeout=0.25, write_timeout=10) as esp:
        time.sleep(0.3)
        esp.reset_input_buffer()
        esp.write(b'HELLO\n')
        print(expect(esp, 'READY PHASE8'))
        esp.write(b'RUN\n')
        go = expect(esp, 'GO ')
        payload = b''.join(chunks)
        assert go == f'GO {len(payload)}', f'Wrong device payload: {go}'
        started = time.perf_counter()
        transferred = 0
        for chunk in chunks:
            esp.write(chunk)
            transferred += len(chunk)
            got = expect(esp, 'ACK ')
            assert got == f'ACK {transferred}', f'ACK mismatch: got {got}, expected ACK {transferred}'
        rows = []
        for token in range(TOKENS):
            response = expect(esp, 'ROW ')
            values = response.split()
            assert int(values[1]) == token, f'Unexpected token row: {response}'
            assert len(values) == 2 + DIM, f'Wrong number of attention components: {response}'
            rows.append([float(value) for value in values[2:]])
        done = expect(esp, 'DONE ')
        elapsed = (time.perf_counter() - started) * 1000

    parts = done.split()
    assert len(parts) == 7, f'Unexpected DONE: {done}'
    _, byte_count, weight_hash, projection_us, attention_us, receive_us, device_us = parts
    assert int(byte_count) == len(payload), 'Byte count mismatch'
    assert int(weight_hash, 16) == fnv32(payload), 'Weight/input hash mismatch'
    observed = np.asarray(rows, dtype=np.float64)
    maximum_error = float(np.max(np.abs(observed - reference)))
    print('Attention output row 3:', np.round(observed[3], 4).tolist())
    print(f'Max absolute difference vs Python reference: {maximum_error:.6f}')
    print(f'USB + full pass: {elapsed:.3f} ms | device elapsed: {int(device_us) / 1000:.3f} ms')
    print(f'ESP32 Q/K/V projections: {int(projection_us) / 1000:.3f} ms')
    print(f'ESP32 attention: {int(attention_us) / 1000:.3f} ms')
    print(f'ESP32 receive/wait: {int(receive_us) / 1000:.3f} ms')
    if maximum_error > 0.03:
        raise AssertionError('Attention outputs differ too much from Python reference')
    print('PASS: transfer hash verified, all 64 attention outputs match Python reference')
    print('PHASE 8 CAUSAL SELF-ATTENTION COMPLETE')


if __name__ == '__main__':
    main()
