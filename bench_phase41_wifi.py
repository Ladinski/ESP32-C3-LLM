"""Phase 41: checksum-validated 4MiB Wi-Fi TCP benchmark; never performs AI inference."""
import argparse
import hashlib
import json
import socket
import statistics
import time
from pathlib import Path

FNV_OFFSET = 2166136261
FNV_PRIME = 16777619

def fnv1a(data):
    h = FNV_OFFSET
    for b in data:
        h = ((h ^ b) * FNV_PRIME) & 0xFFFFFFFF
    return h

def deterministic_bytes(length):
    # Non-trivial reproducible bytes; checksum is computed BEFORE timing.
    seed = hashlib.sha256(b"ESP32 Phase41 TCP-vs-USB transport").digest()
    return (seed * ((length + len(seed) - 1)//len(seed)))[:length]

def line(sock):
    raw=bytearray()
    while len(raw)<256:
        b=sock.recv(1)
        if not b: raise ConnectionError('Wi-Fi connection closed')
        if b == b'\n': return raw.decode('ascii','replace')
        raw += b
    raise ValueError('Firmware response too long')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--host',default='192.168.4.1')
    p.add_argument('--port',type=int,default=3333)
    p.add_argument('--mib',type=int,default=4)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--usb-mib-s',type=float,default=0.764,
                   help='Previously measured USB median MiB/s; for screening, not simultaneous A/B')
    a=p.parse_args()
    if not 1 <= a.mib <= 16 or not 1 <= a.repeats <= 10:
        p.error('mib must be 1..16 and repeats 1..10')
    data=deterministic_bytes(a.mib*1024*1024)
    expected=fnv1a(data)
    print(f'PHASE41 Wi-Fi TCP | {a.mib} MiB x {a.repeats} | FNV {expected:08x}',flush=True)
    speeds=[]; elapsed=[]
    for i in range(a.repeats):
        with socket.create_connection((a.host,a.port),timeout=8) as s:
            s.settimeout(30)
            s.sendall(f'STREAM {len(data)}\n'.encode())
            ready=line(s)
            if ready!='READY PHASE41 WIFI':raise RuntimeError(f'Wrong firmware: {ready}')
            for off in range(0,len(data),4096):
                s.sendall(memoryview(data)[off:off+4096])
            response=line(s)
        parts=response.split()
        if len(parts)!=6 or parts[0]!='WIFI_STATS':
            raise RuntimeError('Missing or unexpected firmware statistics: '+response)
        transferred,fnv,device_us,wait_us,heap=(int(parts[1]),int(parts[2],16),int(parts[3]),int(parts[4]),int(parts[5]))
        if transferred!=len(data) or fnv!=expected:
            raise RuntimeError(f'FAILED DATA INTEGRITY: {response}')
        secs=device_us/1e6
        rate=a.mib/secs
        elapsed.append(secs);speeds.append(rate)
        print(f'  run {i+1}/{a.repeats}: {secs:.3f}s | {rate:.3f} MiB/s | RX wait {wait_us/1e6:.3f}s | free heap {heap}',flush=True)
        time.sleep(.25)
    median=statistics.median(speeds)
    gain=100*(median/a.usb_mib_s-1)
    print(f'Wi-Fi median {median:.3f} MiB/s | reference USB {a.usb_mib_s:.3f} MiB/s | difference {gain:+.1f}%',flush=True)
    if median>=1.5*a.usb_mib_s:
        print('PHASE41 TRANSPORT GATE CANDIDATE: >=50% higher throughput; investigate inference integration, accuracy, and Wi-Fi RAM cost BEFORE adoption.')
    else:
        print('PHASE41 TRANSPORT GATE FAIL: <50% screening improvement. Restore Phase39; no inference integration.')
    results={'transport':'wifi_tcp','mib':a.mib,'times_seconds':elapsed,'rates_mib_s':speeds,
             'median_mib_s':median,'reference_usb_mib_s':a.usb_mib_s,'improvement_pct':gain,
             'data_checksum':f'{expected:08x}'}
    Path('phase41_wifi_results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')

if __name__=='__main__':main()
