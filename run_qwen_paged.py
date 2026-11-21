"""Phase 23: ESP32-C3 Qwen-style kernel test with ESP32-directed 512B weight pages.
The fixture is synthetic, deterministic INT4. Not pretrained Qwen inference.
"""
import argparse
import hashlib
import math
import struct
import time
from pathlib import Path

D, KV, FF, PAGE = 32, 16, 64, 512
MATS = ((32, 32, 1), (16, 32, 2), (16, 32, 3),
        (32, 32, 4), (64, 32, 5), (64, 32, 6), (32, 64, 7))
FNV_OFFSET, FNV_PRIME = 2166136261, 16777619


def fixture():
    data=bytearray()
    for rows, cols, seed in MATS:
        for r in range(rows):
            for c in range(0,cols,2):
                def val(j):
                    return ((r*17+j*13+seed*7+(r*j)%11)%15)-7
                data.append((val(c)&15)|((val(c+1)&15)<<4))
            data+=struct.pack('<f',.009+.001*((r+seed)%7))
    assert len(data)==5632
    return bytes(data)


def expected_storage(image):
    base=0
    seq=[]
    for rows,cols,_ in MATS:
        width=cols//2+4
        for r in range(rows):seq.append((base+r*width,width))
        base+=rows*width
    assert base==len(image)
    reads=0; hits=0; physical=0; last=None; hashval=FNV_OFFSET
    for _ in range(5):
        for off,n in seq:
            record=image[off:off+n]
            for b in record:hashval=((hashval^b)*FNV_PRIME)&0xffffffff
            copied=0
            while copied<n:
                pos=off+copied; page=pos//PAGE*PAGE
                if page==last:hits+=1
                else:reads+=1;physical+=min(PAGE,len(image)-page);last=page
                copied+=min(n-copied,PAGE-pos%PAGE)
    return reads,hits,physical,len(seq)*5,hashval


def write_image(path):
    blob=fixture()
    path.write_bytes(blob)
    print(f'IMAGE PASS: {path} | {len(blob)} bytes | SHA256 {hashlib.sha256(blob).hexdigest()}')
    return blob


def self_test(blob):
    assert blob==fixture(), 'Image differs from deterministic test fixture'
    expect=expected_storage(blob)
    print(f'PAGE SIMULATION PASS: fetches={expect[0]} page hits={expect[1]} transfer={expect[2]} records={expect[3]} FNV32={expect[4]:08x}')
    from test_qwen_kernels import reference
    y=reference()
    assert y.shape==(5,32) and math.isfinite(float(y.sum()))
    print('REFERENCE PASS: five positions, 32 features; test fixture only')


def run_hardware(blob,port,baud,timeout,storage_only):
    import serial
    from serial.serialutil import SerialException
    if not storage_only:
        from test_qwen_kernels import check_rows
    deadline=time.monotonic()+timeout
    last_start=0.0
    dev=None; rows={}; ready=False; done=False; stats=None; reqs=0; nbytes=0
    try:
        while time.monotonic()<deadline and not done:
            try:
                if dev is None:
                    dev=serial.Serial(port,baudrate=baud,timeout=.3,write_timeout=2)
                    time.sleep(.6) # native USB serial/JTAG can reset/re-enumerate on open
                if not ready and time.monotonic()-last_start>1.5:
                    dev.write(b'START\n');last_start=time.monotonic()
                raw=dev.readline()
                if not raw:continue
                line=raw.decode('ascii',errors='replace').strip()
                if not line:continue
                if line=='READY PHASE23':
                    print(line);ready=True;continue
                if not ready:continue
                if line.startswith('PAGE '):
                    fields=line.split()
                    if len(fields)!=3:raise RuntimeError('Malformed PAGE: '+line)
                    offset,count=map(int,fields[1:])
                    if offset%PAGE or not 0<count<=PAGE or offset+count>len(blob) or count!=min(PAGE,len(blob)-offset):
                        raise RuntimeError('Out-of-bounds/nonaligned ESP32 storage request: '+line)
                    dev.write(blob[offset:offset+count]);reqs+=1;nbytes+=count
                    continue
                if line.startswith('OUT '):
                    parts=line.split(); pos=int(parts[1]);values=[float(x) for x in parts[2:]]
                    if pos in rows or pos not in range(5) or len(values)!=32 or not all(map(math.isfinite,values)):
                        raise RuntimeError('Invalid device activation: '+line)
                    rows[pos]=values;print(f'OUT {pos} received (32 ESP32-computed values)');continue
                if line.startswith('STATS '):
                    fields=line.split()
                    if len(fields)!=8:raise RuntimeError('Bad STATS: '+line)
                    stats=tuple(map(int,fields[1:5]))+(int(fields[5],16),)+tuple(map(int,fields[6:]))
                    print(line);continue
                if line.startswith('DONE PHASE23'):
                    done=True;print(line);break
                if line.startswith('ERR '):raise RuntimeError('ESP32 error: '+line)
                if line.startswith('CONFIG '):print(line)
            except (SerialException, OSError) as exc:
                if ready:
                    raise RuntimeError('Serial connection lost mid-inference; close monitor and retry') from exc
                print(f'USB reconnect: {exc}')
                if dev:
                    try:dev.close()
                    except Exception:pass
                dev=None;time.sleep(.3)
        if not done:raise TimeoutError(f'No DONE PHASE23 in {timeout}s; ready={ready}, rows={sorted(rows)}')
        if sorted(rows)!=list(range(5)):raise RuntimeError('Incomplete neural output')
        expected=expected_storage(blob)
        if stats is None or stats[:5]!=expected:
            raise RuntimeError(f'ESP32 storage stats {stats} != independent Python model {expected}')
        if (reqs,nbytes)!=(expected[0],expected[2]):
            raise RuntimeError(f'Physical bytes/requests {(reqs,nbytes)} != {(expected[0],expected[2])}')
        if stats[5]<65536 or stats[6]<256:raise RuntimeError('Insufficient ESP32 heap/stack headroom')
        print(f'STORAGE PASS: {reqs} ESP32-selected 512B pages; {nbytes} physical bytes; {expected[3]} INT4 row records')
        if not storage_only:check_rows(rows)
        else:print('STORAGE-ONLY PASS: PC served bytes without executing any neural layers')
        print('PHASE 23 PASS: ESP32-owned paged INT4 weights + Qwen-style neural kernels + persistent GQA KV cache')
    finally:
        if dev:dev.close()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--image',default='phase23_qwen_fixture.bin')
    ap.add_argument('--port',default='COM3')
    ap.add_argument('--baud',type=int,default=115200)
    ap.add_argument('--timeout',type=float,default=90)
    ap.add_argument('--self-test',action='store_true')
    ap.add_argument('--storage-only',action='store_true',help='PC only serves bytes: no Python reference calculations')
    a=ap.parse_args()
    path=Path(a.image)
    if not path.exists():write_image(path)
    blob=path.read_bytes()
    if blob!=fixture():raise RuntimeError('Model fixture corrupted: remove file and regenerate')
    print('PHASE 23 | ESP32-directed paged Qwen-style INT4 kernels | SYNTHETIC fixture')
    if a.self_test:self_test(blob)
    else:run_hardware(blob,a.port,a.baud,a.timeout,a.storage_only)

if __name__=='__main__':main()
