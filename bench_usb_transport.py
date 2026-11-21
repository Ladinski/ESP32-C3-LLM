"""Phase 31: ESP32-owned 4 KiB read benchmark, with no neural computation.

This checks serial delivery and separates Python write-blocking time from
ESP32 time waiting for bytes. Host performs no transformer inference.
"""
import argparse
import hashlib
import json
import mmap
import time
from pathlib import Path

BLOCK = 4096
FNV_OFFSET = 2166136261
FNV_PRIME = 16777619


def checksum(data):
    value = FNV_OFFSET
    for byte in data:
        value = ((value ^ byte) * FNV_PRIME) & 0xFFFFFFFF
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', default='phase25_qwen_24layers_int4.bin')
    parser.add_argument('--metadata', default='phase25_qwen_24layers_int4.json')
    parser.add_argument('--port', default='COM3')
    parser.add_argument('--baud', type=int, default=115200)
    parser.add_argument('--mib', type=int, default=4, help='1 to 16 MiB of consecutive model bytes')
    parser.add_argument('--write-chunk', type=int, choices=(256, 512, 1024, 2048, 4096), default=4096,
                        help='Max size of each individual Python serial write')
    parser.add_argument('--timeout', type=int, default=180)
    args = parser.parse_args()
    if not 1 <= args.mib <= 16:
        parser.error('--mib must be between 1 and 16')
    image = Path(args.image)
    meta = json.loads(Path(args.metadata).read_text(encoding='utf-8'))
    if meta.get('image_size') != image.stat().st_size:
        raise RuntimeError('Model image/metadata size mismatch')
    total = args.mib * 1024 * 1024
    if total > image.stat().st_size:
        raise RuntimeError('Benchmark longer than model image')
    # Precompute independently BEFORE opening serial; avoid stalling the board.
    with image.open('rb') as handle:
        mm = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            expected = checksum(mm[:total])
            import serial
            from serial.serialutil import SerialException
            device = None
            deadline = time.monotonic() + args.timeout
            last_start = 0.0
            ready = False
            begun = False
            count = 0
            physical = 0
            write_elapsed = 0.0
            max_write = 0.0
            profile = None
            started_at = time.perf_counter()
            try:
                while time.monotonic() < deadline:
                    try:
                        if device is None:
                            device = serial.Serial(args.port, args.baud, timeout=0.25, write_timeout=8)
                            time.sleep(0.7)
                        if not ready and time.monotonic() - last_start > 1.5:
                            device.write(f'START BENCH {total}\n'.encode('ascii'))
                            last_start = time.monotonic()
                        raw = device.readline()
                        if not raw:
                            continue
                        line = raw.decode('ascii', errors='replace').strip()
                        if line == 'READY PHASE31':
                            ready = True
                            print(line, flush=True)
                        elif not ready:
                            continue
                        elif line == f'BLOCK_BYTES {BLOCK}':
                            print(line, flush=True)
                        elif line.startswith('BLOCK_BYTES '):
                            raise RuntimeError(f'Unexpected firmware block size: {line}')
                        elif line.startswith('BENCH_BEGIN '):
                            if int(line.split()[1]) != total:
                                raise RuntimeError('Firmware benchmark length differs')
                            begun = True
                            print(line, flush=True)
                        elif line.startswith('PAGE '):
                            if not begun:
                                raise RuntimeError('PAGE arrived before BENCH_BEGIN')
                            _, offset_text, n_text = line.split()
                            offset, n = int(offset_text), int(n_text)
                            if offset != physical or n != min(BLOCK, len(mm) - offset):
                                raise RuntimeError('Unexpected ESP32-requested model address: ' + line)
                            for pos in range(offset, offset + n, args.write_chunk):
                                piece = mm[pos:min(pos + args.write_chunk, offset + n)]
                                before = time.perf_counter()
                                sent = device.write(piece)
                                dt = time.perf_counter() - before
                                write_elapsed += dt
                                max_write = max(max_write, dt)
                                if sent != len(piece):
                                    raise RuntimeError('Partial serial write')
                            physical += n
                            count += 1
                            if count % 256 == 0:
                                print(f'  ESP32 requested {count:,} blocks; {physical / (1024*1024):.1f} MiB served', flush=True)
                        elif line.startswith('BENCH_STATS '):
                            parts = line.split()
                            if len(parts) != 9:
                                raise RuntimeError('Unexpected BENCH_STATS format: ' + line)
                            nreads, nbytes = map(int, parts[1:3])
                            received_hash = int(parts[3], 16)
                            request_us, receive_us, elapsed_us, heap, stack = map(int, parts[4:])
                            if (nreads, nbytes, received_hash) != (count, physical, expected):
                                raise RuntimeError(f'Transport mismatch: device={nreads,nbytes,hex(received_hash)}, expected={count,physical,hex(expected)}')
                            if stack < 256 or heap < 32768:
                                raise RuntimeError('Insufficient memory headroom')
                            profile = (request_us, receive_us, elapsed_us, heap, stack)
                            print(f'BENCH STORAGE PASS: {nreads:,} blocks; {nbytes:,} bytes; FNV32 {received_hash:08x}', flush=True)
                        elif line == 'DONE PHASE31 BENCH':
                            if profile is None or physical != total:
                                raise RuntimeError('Incomplete benchmark')
                            wall = time.perf_counter() - started_at
                            req, wait, elapsed, heap, stack = profile
                            print(f'ESP32 total: {elapsed/1e6:.3f}s; request sending: {req/1e6:.3f}s; waiting for data: {wait/1e6:.3f}s', flush=True)
                            print(f'Python serial write blocking: {write_elapsed:.3f}s; slowest write: {max_write*1000:.1f}ms', flush=True)
                            print(f'Effective ESP32 transfer throughput: {(total/(1024*1024))/(elapsed/1e6):.3f} MiB/s', flush=True)
                            print(f'Host wall time including connection: {wall:.3f}s; heap={heap} bytes; stack headroom={stack} bytes', flush=True)
                            print(f'PHASE 30 TRANSPORT BENCH PASS | block={BLOCK} B | write chunk={args.write_chunk} B', flush=True)
                            return
                        elif line.startswith('ERR '):
                            raise RuntimeError('ESP32 reported: ' + line)
                    except (SerialException, OSError) as exc:
                        if ready:
                            raise RuntimeError('COM port disconnected during benchmark') from exc
                        if device is not None:
                            try:
                                device.close()
                            except OSError:
                                pass
                        device = None
                        time.sleep(0.4)
                raise TimeoutError('Transport benchmark timed out; close idf.py monitor and confirm firmware was flashed')
            finally:
                if device is not None:
                    device.close()
        finally:
            mm.close()


if __name__ == '__main__':
    main()
