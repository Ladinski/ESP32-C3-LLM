"""Device prompts with exact-logit Python verification; no PC-computed token fed back."""
import argparse
import time
from pathlib import Path
import numpy as np
from phase12_common import CHARS, VOCAB, MAX_TOKENS, fnv32, unpack_model, prompt_decode, full_prefix_decode, expect


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', default='COM3')
    ap.add_argument('--prompt', default='hi ')
    ap.add_argument('--steps', type=int, default=12)
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args()
    if not args.prompt or any(ch not in CHARS for ch in args.prompt):
        ap.error(f'Prompt must contain only characters in {CHARS!r}')
    if args.steps < 1 or len(args.prompt) + args.steps - 1 > MAX_TOKENS:
        ap.error(f'prompt_length + steps - 1 must be <= {MAX_TOKENS}')
    payload = (Path(__file__).resolve().parent / 'phase12_model.bin').read_bytes()
    emb, ws = unpack_model(payload)
    pt = [CHARS.index(ch) for ch in args.prompt]
    seen, generated, reference = prompt_decode(emb, ws, pt, args.steps)
    full = full_prefix_decode(emb, ws, seen)[len(pt)-1:]
    difference = float(np.max(np.abs(reference - full)))
    assert difference < 1e-10, f'PC cache/full-prefix mismatch {difference}'
    print('ESP32-C3 PHASE 12 | MULTI-TEXT trained transformer with prompt prefill')
    print(f'Model {len(payload)} bytes | FNV32 {fnv32(payload):08x} | prompt {args.prompt!r} | generate {args.steps}')
    print('Python cached/full-prefix error:', f'{difference:.3g}')
    if args.self_test:
        print('PC reference generation:', repr(args.prompt + ''.join(CHARS[t] for t in generated)))
        print('SELF-TEST PASS')
        return
    import serial
    with serial.Serial(args.port, 115200, timeout=.25, write_timeout=10) as esp:
        time.sleep(.3)
        esp.reset_input_buffer()
        esp.write(b'HELLO\n')
        print(expect(esp, 'READY PHASE12'))
        esp.write(b'LOAD\n')
        assert expect(esp, 'GO ') == f'GO {len(payload)}'
        for i in range(0, len(payload), 256):
            chunk = payload[i:i + 256]
            esp.write(chunk)
            assert expect(esp, 'ACK ') == f'ACK {i + len(chunk)}'
        loaded = expect(esp, 'LOADED ').split()
        assert int(loaded[1]) == len(payload) and int(loaded[2], 16) == fnv32(payload)
        print('PASS: model bytes and hash verified')
        cmd = f'PROMPT {len(pt)} {args.steps} ' + ' '.join(str(t) for t in pt) + '\n'
        esp.write(cmd.encode('ascii'))
        started = time.perf_counter()
        actual = []
        for t in range(args.steps):
            line = expect(esp, 'SCORES ').split()
            assert len(line) == VOCAB + 2 and int(line[1]) == t, line
            actual.append([float(v) for v in line[2:]])
            step = expect(esp, 'STEP ').split()
            assert len(step) == 4 and int(step[1]) == t
            expected_input = seen[len(pt) - 1 + t]
            assert int(step[2]) == expected_input, (step, expected_input)
            assert int(step[3]) == generated[t], (step, generated[t])
            print(f'[{t+1:2}/{args.steps}] {CHARS[int(step[2])]!r} -> {CHARS[int(step[3])]!r} PASS')
        finished = expect(esp, 'DONE ').split()
        wall_ms = (time.perf_counter() - started) * 1000
    assert len(finished) == 6 and int(finished[1]) == args.steps
    assert int(finished[2], 16) == fnv32(payload)
    err = float(np.max(np.abs(np.asarray(actual) - reference)))
    print('Max device-vs-PC logit error:', f'{err:.6f}')
    print('USB + prompt prefill + decoding:', f'{wall_ms:.3f} ms')
    print('ESP32 elapsed:', f'{int(finished[5])/1000:.3f} ms')
    assert err < .003, f'Device score mismatch {err}'
    print('Generated:', repr(args.prompt + ''.join(CHARS[t] for t in generated)))
    print('PHASE 12 PASS: prompt prefill, persistent KV cache and all logits match Python')


if __name__ == '__main__':
    main()
