"""Phase 40. Repeat 4-MiB real model-byte USB streams and compare medians.

No neural computation on the PC; no model decoding on the ESP32 for this test.
The same 4 KiB writes and on-device FNV verification are used in both cases.
"""
import argparse
import json
import mmap
import statistics
import time
from pathlib import Path

FNV_INIT = 2166136261
FNV_PRIME = 16777619
MiB = 1024 * 1024


def fnv32(data):
    digest = FNV_INIT
    for byte in data:
        digest = ((digest ^ byte) * FNV_PRIME) & 0xffffffff
    return digest


def run_once(args, mapped, expected_hash, total):
    import serial
    from serial.serialutil import SerialException

    deadline = time.monotonic() + args.timeout
    last_sent = 0.0
    dev = None
    ready = False
    streaming = False
    sent = 0
    stats = None
    write_seconds = 0.0
    try:
        while time.monotonic() < deadline:
            try:
                if dev is None:
                    dev = serial.Serial(args.port, args.baud, timeout=0.25, write_timeout=10)
                    time.sleep(0.45)
                if not ready and time.monotonic() - last_sent > 1.0:
                    dev.write(f'START STREAM {total}\n'.encode('ascii'))
                    last_sent = time.monotonic()
                raw = dev.readline()
                if not raw:
                    continue
                line = raw.decode('ascii', errors='replace').strip()
                if line in ('READY PHASE33', 'READY PHASE40'):
                    ready = True
                elif not ready:
                    continue
                elif line.startswith('BLOCK_BYTES '):
                    if int(line.split()[1]) != 4096:
                        raise RuntimeError('Unexpected inference block size: ' + line)
                elif line.startswith('STREAM_READY '):
                    if streaming or int(line.split()[1]) != total:
                        raise RuntimeError('Unexpected or duplicate stream request: ' + line)
                    streaming = True
                    for offset in range(0, total, 4096):
                        chunk = mapped[offset:offset + 4096]
                        started = time.perf_counter()
                        n = dev.write(chunk)
                        write_seconds += time.perf_counter() - started
                        if n != len(chunk):
                            raise RuntimeError('Short write')
                        sent += n
                elif line.startswith('STREAM_STATS '):
                    parts = line.split()
                    if len(parts) != 8:
                        raise RuntimeError('Malformed device results: ' + line)
                    chunks = int(parts[1]); nbytes = int(parts[2]); digest = int(parts[3], 16)
                    rx_us, total_us, heap, stack = map(int, parts[4:])
                    if chunks != total // 4096 or nbytes != total or digest != expected_hash:
                        raise RuntimeError('ON-DEVICE MODEL BYTES/CHECKSUM MISMATCH')
                    if heap < 32768 or stack < 256:
                        raise RuntimeError('Heap/stack safety check failed')
                    stats = {'seconds': total_us / 1e6,
                             'mib_per_s': (total / MiB) / (total_us / 1e6),
                             'rx_seconds': rx_us / 1e6, 'host_write_seconds': write_seconds,
                             'heap': heap, 'stack': stack}
                elif line == 'DONE PHASE33 STREAM':
                    if stats is None or sent != total:
                        raise RuntimeError('Device completed without full validated stream')
                    return stats
                elif line.startswith('ERR '):
                    raise RuntimeError('ESP32 reported: ' + line)
            except (SerialException, OSError) as exc:
                if ready:
                    raise RuntimeError('USB disconnected mid-stream') from exc
                if dev is not None:
                    try: dev.close()
                    except OSError: pass
                dev = None
                time.sleep(.3)
        raise TimeoutError('No complete transfer; check COM3 and close the serial monitor')
    finally:
        if dev is not None:
            dev.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port', default='COM3')
    p.add_argument('--baud', type=int, default=115200)
    p.add_argument('--image', default='phase25_qwen_24layers_int4.bin')
    p.add_argument('--metadata', default='phase25_qwen_24layers_int4.json')
    p.add_argument('--mib', type=int, default=4)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--timeout', type=float, default=45)
    modes = p.add_mutually_exclusive_group(required=True)
    modes.add_argument('--baseline', action='store_true', help='Record Phase 39 / 8 KB RX baseline')
    modes.add_argument('--candidate', action='store_true', help='Compare Phase 40 / 16 KB RX against baseline')
    p.add_argument('--results', default='phase40_baseline.json')
    a = p.parse_args()
    if not 1 <= a.mib <= 16 or not 2 <= a.repeats <= 7:
        p.error('Choose 1..16 MiB and 2..7 repetitions')
    image = Path(a.image)
    metadata = json.loads(Path(a.metadata).read_text(encoding='utf-8'))
    if metadata.get('image_size') != image.stat().st_size:
        raise RuntimeError('Model image size mismatch')
    nbytes = a.mib * MiB
    if nbytes > image.stat().st_size:
        raise RuntimeError('Requested more bytes than model contains')
    results_path = Path(a.results)
    baseline = None
    if a.candidate:
        baseline = json.loads(results_path.read_text(encoding='utf-8'))
        if baseline.get('mode') != 'baseline' or baseline.get('bytes') != nbytes or baseline.get('repeats') != a.repeats:
            raise RuntimeError('Baseline comparison settings do not match')
    with image.open('rb') as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        digest = fnv32(mm[:nbytes])
        runs = []
        print(f'PHASE40 {"BASELINE 8KB_RX" if a.baseline else "CANDIDATE 16KB_RX"} | {a.mib} MiB | {a.repeats} repeats | 4096-byte writes | FNV {digest:08x}', flush=True)
        for attempt in range(a.repeats):
            stats = run_once(a, mm, digest, nbytes)
            runs.append(stats)
            print(f'  run {attempt+1}/{a.repeats}: {stats["seconds"]:.3f}s | {stats["mib_per_s"]:.3f} MiB/s | RX wait {stats["rx_seconds"]:.3f}s | heap {stats["heap"]}', flush=True)
    elapsed_median = statistics.median(r['seconds'] for r in runs)
    result = {'mode': 'baseline' if a.baseline else 'candidate', 'bytes': nbytes,
              'repeats': a.repeats, 'median_seconds': elapsed_median, 'runs': runs}
    print(f'MEDIAN DEVICE STREAM {elapsed_median:.3f}s | {a.mib/elapsed_median:.3f} MiB/s', flush=True)
    if a.baseline:
        results_path.write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(f'BASELINE SAVED: {results_path}', flush=True)
        print('Next: flash the Phase 40 candidate, then run --candidate.', flush=True)
    else:
        baseline_seconds = baseline['median_seconds']
        speedup_pct = 100 * (baseline_seconds / elapsed_median - 1.0)
        print(f'BASELINE MEDIAN {baseline_seconds:.3f}s | CANDIDATE MEDIAN {elapsed_median:.3f}s | THROUGHPUT GAIN {speedup_pct:+.2f}%', flush=True)
        if speedup_pct >= 10.0:
            print('PHASE40 TRANSFER GATE PASS: >=10% measured gain. Optional two-layer inference validation is now justified.', flush=True)
        else:
            print('PHASE40 TRANSFER GATE FAIL: <10% measured gain. Restore Phase 39 firmware; DO NOT run inference.', flush=True)


if __name__ == '__main__':
    main()
