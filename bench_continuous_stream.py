"""Phase 31 continuous USB transport test, without host/device inference.

Compared to bench_usb_transport.py (PAGE requests), this test allows the C3
one bounded sequential stream. The PC supplies immutable model bytes only.
"""
import argparse
import json
import mmap
import time
from pathlib import Path

FNV_OFFSET=2166136261
FNV_PRIME=16777619
BLOCK=4096

def checksum(data):
    value=FNV_OFFSET
    for byte in data:
        value=((value^byte)*FNV_PRIME)&0xffffffff
    return value

def run():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image',default='phase25_qwen_24layers_int4.bin')
    p.add_argument('--metadata',default='phase25_qwen_24layers_int4.json')
    p.add_argument('--port',default='COM3')
    p.add_argument('--baud',type=int,default=115200)
    p.add_argument('--mib',type=int,default=4)
    p.add_argument('--write-chunk',type=int,choices=(256,512,1024,2048,4096),default=4096)
    p.add_argument('--timeout',type=float,default=180)
    args=p.parse_args()
    if not 1<=args.mib<=16:p.error('--mib must be 1..16')
    image=Path(args.image)
    metadata=json.loads(Path(args.metadata).read_text(encoding='utf-8'))
    if metadata.get('image_size')!=image.stat().st_size:raise RuntimeError('Image metadata size mismatch')
    total=args.mib*1024*1024
    if total>image.stat().st_size:raise RuntimeError('Too many requested bytes')
    with image.open('rb') as f:
      with mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as mm:
        expected_hash=checksum(mm[:total])  # precomputed before opening USB port
        import serial
        from serial.serialutil import SerialException
        dev=None;ready=False;streaming=False;last_start=0.0
        write_time=0.0;slowest=0.0;sent=0;stats=None
        deadline=time.monotonic()+args.timeout
        wall_start=time.perf_counter()
        try:
          while time.monotonic()<deadline:
            try:
              if dev is None:
                dev=serial.Serial(args.port,args.baud,timeout=0.25,write_timeout=10)
                time.sleep(.7)
              if not ready and time.monotonic()-last_start>1.5:
                dev.write(f'START STREAM {total}\n'.encode())
                last_start=time.monotonic()
              raw=dev.readline()
              if not raw:continue
              line=raw.decode('ascii',errors='replace').strip()
              if line=='READY PHASE31':
                ready=True;print(line,flush=True)
              elif not ready:continue
              elif line.startswith('BLOCK_BYTES '):
                if int(line.split()[1])!=BLOCK:raise RuntimeError('Firmware must have 4 KB buffer: '+line)
                print(line,flush=True)
              elif line.startswith('STREAM_READY '):
                if streaming:raise RuntimeError('Duplicate STREAM_READY')
                n=int(line.split()[1]);
                if n!=total:raise RuntimeError('ESP32 stream length mismatch')
                streaming=True
                print(line,flush=True)
                # Continuous serial writes, NO per-page address/acknowledgment.
                # USB device/driver backpressure bounds outstanding data.
                for offset in range(0,total,args.write_chunk):
                  part=mm[offset:min(offset+args.write_chunk,total)]
                  t=time.perf_counter();count=dev.write(part);dt=time.perf_counter()-t
                  write_time+=dt;slowest=max(slowest,dt)
                  if count!=len(part):raise RuntimeError('Partial USB write')
                  sent+=count
                  if sent%(1024*1024)==0:
                    print(f'  Streamed {sent//(1024*1024)} MiB',flush=True)
              elif line.startswith('STREAM_STATS '):
                v=line.split()
                if len(v)!=8:raise RuntimeError('Bad stream stats '+line)
                chunks,nbytes=int(v[1]),int(v[2]);digest=int(v[3],16)
                rx_us,total_us,heap,stack=map(int,v[4:])
                if (chunks,nbytes,digest)!=(total//BLOCK,total,expected_hash):
                  raise RuntimeError('Stream byte count/FNV mismatch')
                if heap<32768 or stack<256:raise RuntimeError('Low heap or stack')
                stats=(chunks,nbytes,rx_us,total_us,heap,stack)
                print(f'STREAM STORAGE PASS: {chunks:,} chunks, {nbytes:,} bytes, FNV32 {digest:08x}',flush=True)
              elif line=='DONE PHASE31 STREAM':
                if stats is None or sent!=total:raise RuntimeError('Stream did not complete')
                chunks,nbytes,rx_us,total_us,heap,stack=stats
                print(f'ESP32 stream: {total_us/1e6:.3f}s; USB read time: {rx_us/1e6:.3f}s',flush=True)
                print(f'Python serial.write blocking: {write_time:.3f}s; slowest: {slowest*1e3:.1f}ms',flush=True)
                print(f'Effective throughput: {args.mib/(total_us/1e6):.3f} MiB/s',flush=True)
                print(f'Host elapsed including startup: {time.perf_counter()-wall_start:.3f}s; heap={heap}; stack={stack}',flush=True)
                print('PHASE 31 CONTINUOUS STREAM BENCH PASS',flush=True)
                return
              elif line.startswith('ERR '):raise RuntimeError('ESP32: '+line)
            except (SerialException,OSError) as exc:
              if ready:raise RuntimeError('Serial disconnected mid-stream') from exc
              if dev is not None:
                try:dev.close()
                except OSError:pass
              dev=None;time.sleep(.4)
          raise TimeoutError('Stream benchmark timed out; close monitor, verify COM3 and firmware')
        finally:
          if dev is not None:dev.close()
if __name__=='__main__':run()
