"""Phase 61 ESP32-C3 fused dual-Q15 INT4 microbenchmark, USB-C serial only.

This test does not stream pretrained weights or enable the candidate in inference.
"""
import argparse
import re
import serial
import sys
import time

RESULT = re.compile(
    r'^PHASE61_KERNEL valid=(\d+) group=(\d+) outlier=(\d+) baseline_us=(\d+) candidate_us=(\d+) '
    r'base_error=([0-9eE.+-]+) candidate_error=([0-9eE.+-]+) selected=(BENCH_PASS|FALLBACK)$'
)


def run(port: str, baud: int, timeout: float) -> str:
    with serial.Serial(port, baudrate=baud, timeout=0.5, write_timeout=3) as dev:
        time.sleep(3)
        dev.reset_input_buffer()
        dev.write(b'START P61 BENCH\n')
        dev.flush()
        print("SENT 'START P61 BENCH'", flush=True)
        deadline = time.monotonic() + timeout
        ready = done = False
        parsed = None
        while time.monotonic() < deadline:
            line = dev.readline().decode('utf-8', 'replace').strip()
            if not line:
                continue
            print(line, flush=True)
            if line == 'READY PHASE61 BENCH':
                ready = True
            if (match := RESULT.fullmatch(line)):
                parsed = match
            if line == 'DONE PHASE61 BENCH':
                done = True
                break
        if not (ready and done and parsed):
            raise RuntimeError('Phase 61 protocol incomplete; do not run inference.')
        valid, group, outlier, old_us, new_us, base_err, candidate_err, selected = parsed.groups()
        old_us, new_us = int(old_us), int(new_us)
        if old_us <= 0 or new_us <= 0:
            raise RuntimeError('Invalid timing values')
        if selected == 'BENCH_PASS' and (valid != '1' or new_us > old_us or float(candidate_err) > float(base_err) + 0.0001):
            raise RuntimeError('Firmware reported pass violating integrity/speed gates')
        print(f'ROW KERNEL: baseline={old_us / 1000:.3f} ms, peeled={new_us / 1000:.3f} ms, change={100 * (new_us / old_us - 1):+.2f}%')
        if selected == 'BENCH_PASS':
            print('PHASE61 BENCH PASS: fused precision path is a candidate. Do not enable production inference yet.')
        else:
            print('PHASE61 FALLBACK: retain original SHIFT + grouped Q15; no full-model test.')
        return selected


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--port', default='COM3')
    ap.add_argument('--baud', type=int, default=115200)
    ap.add_argument('--timeout', type=float, default=35)
    args = ap.parse_args()
    run(args.port, args.baud, args.timeout)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'PHASE61 ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
