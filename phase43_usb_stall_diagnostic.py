"""Phase 43: read-only 4-KiB USB-C streaming stall diagnostic.

Uses the EXISTING Phase 39 firmware's START STREAM command. Does not change
weights, firmware, connection, or run neural arithmetic on the PC.
"""
import argparse
import hashlib
import json
import mmap
from pathlib import Path
import statistics
import time

MIB = 1024 * 1024
FNV0 = 2166136261
FNV_PRIME = 16777619


def fnv32(data):
    h = FNV0
    for byte in data:
        h = ((h ^ byte) * FNV_PRIME) & 0xffffffff
    return h


def percentile(values, fraction):
    if not values:
        return 0.0
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lo = int(position)
    hi = min(len(values) - 1, lo + 1)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def run_once(args, mapped, expected_hash, total):
    import serial
    from serial.serialutil import SerialException
    deadline = time.monotonic() + args.timeout
    dev = None
    last_start = 0
    ready = False
    streaming = False
    stats = None
    chunk_durations = []
    windows = []
    try:
        while time.monotonic() < deadline:
            try:
                if dev is None:
                    dev = serial.Serial(args.port, baudrate=args.baud, timeout=.25, write_timeout=10)
                    time.sleep(.45)
                if not ready and time.monotonic() - last_start > 1.0:
                    dev.write(f'START STREAM {total}\n'.encode('ascii'))
                    last_start = time.monotonic()
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
                        raise RuntimeError('Wrong firmware block size: ' + line)
                elif line.startswith('STREAM_READY '):
                    if streaming or int(line.split()[1]) != total:
                        raise RuntimeError('Invalid duplicate/size in STREAM_READY: ' + line)
                    streaming = True
                    window_bytes = 0
                    window_started = time.perf_counter()
                    for off in range(0, total, 4096):
                        chunk = mapped[off:off+4096]
                        start = time.perf_counter()
                        written = dev.write(chunk)
                        finish = time.perf_counter()
                        if written != len(chunk):
                            raise RuntimeError('Short serial write')
                        chunk_durations.append(finish-start)
                        window_bytes += written
                        if window_bytes >= args.window_kib*1024 or off+4096 == total:
                            dt = finish-window_started
                            windows.append({'mib_per_s': window_bytes/MIB/dt, 'seconds': dt, 'bytes': window_bytes})
                            window_bytes = 0
                            window_started = finish
                elif line.startswith('STREAM_STATS '):
                    parts = line.split()
                    if len(parts) != 8:
                        raise RuntimeError('Malformed STREAM_STATS: ' + line)
                    chunks, received = map(int, parts[1:3])
                    digest = int(parts[3], 16)
                    rx_us, total_us, heap, stack = map(int, parts[4:])
                    if chunks != total//4096 or received != total or digest != expected_hash:
                        raise RuntimeError('MODEL BYTE INTEGRITY FAILED: ' + line)
                    if heap < 32768 or stack < 256:
                        raise RuntimeError('ESP32 memory safety floor failed')
                    stats = {'seconds': total_us/1e6, 'rx_wait_seconds': rx_us/1e6,
                             'mib_per_s': total/MIB/(total_us/1e6), 'heap': heap, 'stack': stack}
                elif line == 'DONE PHASE33 STREAM':
                    if stats is None or not streaming or len(chunk_durations) != total//4096:
                        raise RuntimeError('Missing completed/checksummed USB data')
                    dt = chunk_durations
                    sample_ms = [x*1000 for x in dt]
                    stats.update({
                        'host_write_seconds': sum(dt),
                        'write_ms_p50': percentile(sample_ms,.5),
                        'write_ms_p95': percentile(sample_ms,.95),
                        'write_ms_p99': percentile(sample_ms,.99),
                        'write_ms_max': max(sample_ms),
                        'write_gt_10ms': sum(x > 10 for x in sample_ms),
                        'write_gt_20ms': sum(x > 20 for x in sample_ms),
                        'window_speed_min': min(w['mib_per_s'] for w in windows),
                        'window_speed_max': max(w['mib_per_s'] for w in windows),
                        'window_speed_p50': percentile([w['mib_per_s'] for w in windows], .5),
                        'windows': windows,
                    })
                    return stats
                elif line.startswith('ERR '):
                    raise RuntimeError('ESP32 firmware failure: ' + line)
            except (SerialException, OSError) as exc:
                if ready:
                    raise RuntimeError('USB disconnected mid-stream') from exc
                if dev is not None:
                    try:
                        dev.close()
                    except OSError:
                        pass
                dev = None
                time.sleep(.3)
        raise TimeoutError('No completed test; monitor must be closed and Phase 39 firmware restored')
    finally:
        if dev is not None:
            dev.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', default='COM3')
    parser.add_argument('--baud', type=int, default=115200)
    parser.add_argument('--image', default='phase25_qwen_24layers_int4.bin')
    parser.add_argument('--metadata', default='phase25_qwen_24layers_int4.json')
    parser.add_argument('--mib', type=int, default=4)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--window-kib', type=int, default=256)
    parser.add_argument('--timeout', type=float, default=60)
    parser.add_argument('--report', default='phase43_usb_stall_report.json')
    a = parser.parse_args()
    if not 1 <= a.mib <= 16 or not 1 <= a.repeats <= 6:
        parser.error('--mib must be 1..16 and --repeats 1..6')
    if a.window_kib not in (64,128,256,512,1024):
        parser.error('--window-kib must be 64, 128, 256, 512 or 1024')
    path = Path(a.image)
    metadata = json.loads(Path(a.metadata).read_text(encoding='utf-8'))
    if path.stat().st_size != metadata.get('image_size'):
        raise RuntimeError('Model image size mismatch')
    total = a.mib*MIB
    if total > path.stat().st_size:
        raise RuntimeError('Requested beyond model image')
    with path.open('rb') as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        expected_hash = fnv32(mm[:total]); results = []
        print(f'PHASE 43 USB-C STALL DIAGNOSTIC | {a.mib} MiB x {a.repeats} | 4KiB writes | FNV {expected_hash:08x}',flush=True)
        for i in range(a.repeats):
            r = run_once(a,mm,expected_hash,total)
            results.append(r)
            print(f'  run {i+1}: {r["seconds"]:.3f}s | {r["mib_per_s"]:.3f} MiB/s | RX wait {r["rx_wait_seconds"]:.3f}s | host write p95 {r["write_ms_p95"]:.2f}ms, p99 {r["write_ms_p99"]:.2f}ms, max {r["write_ms_max"]:.2f}ms | >10ms {r["write_gt_10ms"]} | window {r["window_speed_min"]:.3f}-{r["window_speed_max"]:.3f} MiB/s',flush=True)
    speeds = [r['mib_per_s'] for r in results]
    median = statistics.median(speeds)
    slow = min(speeds)
    fast = max(speeds)
    spread = (fast/slow-1)*100
    print(f'MEDIAN {median:.3f} MiB/s | SPREAD {spread:.1f}% | ALL TRANSFERS CHECKSUM PASS',flush=True)
    if spread > 10:
        print('DIAGNOSIS: >10% run-to-run speed variability. Avoid promoting small transport gains from single runs.',flush=True)
    else:
        print('DIAGNOSIS: <=10% run-to-run variation in this sample. Compare stall distributions before changing transport.',flush=True)
    if max(r['write_gt_10ms'] for r in results) > 0:
        print('OBSERVATION: host write() calls sometimes block >10ms; this does not by itself identify Windows, USB driver, or device backpressure.',flush=True)
    else:
        print('OBSERVATION: no host write() stalls >10ms in these samples. Device RX timing remains the separate bottleneck metric.',flush=True)
    Path(a.report).write_text(json.dumps({'phase':43,'bytes_per_run':total,'runs':results,'median_mib_per_s':median,'spread_pct':spread},indent=2),encoding='utf-8')
    print(f'PHASE 43 PASS: no firmware changes, no PC neural computation. REPORT {a.report}',flush=True)

if __name__ == '__main__':
    main()
