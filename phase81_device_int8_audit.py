"""Phase 81: run real pretrained W8A32 projection rows on ESP32-C3 and audit on PC.

The PC supplies already exported immutable row bytes plus a test input vector.
Only the ESP32 performs the audited projection matvec during the live session.
The CPU NumPy reference runs AFTER device results are received.
"""
import argparse
import hashlib
import json
import mmap
import re
import struct
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = PROJECT / 'phase73_instruct' / 'phase80_int8_images' / 'qwen25_instruct_w8_manifest.json'


def source_and_matrix(path, layer, name):
    meta=json.loads(Path(path).read_text(encoding='utf-8'))
    transformer=meta['transformer']
    if transformer['format']!='QW8S-v1':raise ValueError('Unexpected export format')
    layers=transformer['layers']
    if layer<1 or layer>len(layers):raise ValueError('Layer out of range')
    candidates=[r for r in layers[layer-1]['matrices'] if r['name']==name]
    if len(candidates)!=1:raise ValueError(f'Projection {name} not found')
    m=candidates[0]
    if m['row_bytes'] != m['cols']+4:raise ValueError('Row layout mismatch')
    return meta,m


def deterministic_input(cols, seed=81):
    rng=np.random.default_rng(seed)
    return rng.uniform(-1.0,1.0,cols).astype('<f4')


def reference_for_rows(mm, matrix, x):
    # This check runs offline and is never used to feed neural results to ESP32.
    rows,cols=int(matrix['rows']),int(matrix['cols'])
    result=np.empty(rows,dtype=np.float64)
    for i in range(rows):
        off=int(matrix['offset'])+i*(cols+4)
        q=np.frombuffer(mm,dtype=np.int8,count=cols,offset=off).astype(np.float64)
        scale=struct.unpack_from('<f',mm,off+cols)[0]
        result[i]=np.dot(q,x.astype(np.float64))*scale
    return result


def verify_checksum(path,expected):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    if h.hexdigest()!=expected:raise ValueError('INT8 image SHA256 mismatch')
    print('INT8 IMAGE SHA256 VERIFIED', h.hexdigest(),flush=True)


def run(args):
    import serial
    manifest=Path(args.manifest).resolve()
    meta,m=source_and_matrix(manifest,args.layer,args.projection)
    image=manifest.parent/'qwen25_instruct_transformer_w8.bin'
    if not image.is_file():raise FileNotFoundError(image)
    if image.stat().st_size != meta['transformer']['bytes']:raise ValueError('Image size mismatch')
    verify_checksum(image,meta['transformer']['sha256'])
    rows,cols=int(m['rows']),int(m['cols'])
    if rows>4864 or cols>4864:raise ValueError('Outside test firmware bounds')
    x=deterministic_input(cols,args.seed)
    print(f'PHASE81 DEVICE INT8 layer={args.layer} projection={args.projection} rows={rows} cols={cols}',flush=True)
    with open(image,'rb') as f,mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as mm:
        if args.dry_run:
            expected=reference_for_rows(mm,m,x)
            print('PHASE81 PREPARE PASS expected preview',expected[:5].tolist(),flush=True)
            return
        with serial.Serial(args.port,115200,timeout=2,write_timeout=30) as s:
            time.sleep(0.6)
            s.reset_input_buffer()
            s.write(b'PING\n')
            deadline=time.monotonic()+15
            while True:
                if time.monotonic()>deadline:raise TimeoutError('No PONG. Flash Phase81 firmware, close ESP-IDF monitor.')
                line=s.readline().decode('ascii','replace').strip()
                if line=='PONG P81':break
            print('DEVICE CONNECTED',flush=True)
            s.write(f'RUN {rows} {cols}\n'.encode())
            line=s.readline().decode('ascii','replace').strip()
            if line!=f'ACTIVATIONS {cols*4}':raise RuntimeError(f'Expected ACTIVATIONS: {line}')
            s.write(x.tobytes())
            for i in range(rows):
                line=s.readline().decode('ascii','replace').strip()
                if line != f'ROW {i}':raise RuntimeError(f'Expected ROW {i}, got {line}')
                off=int(m['offset'])+i*(cols+4)
                blob=mm[off:off+cols+4]
                if len(blob)!=cols+4:raise RuntimeError('Truncated row in image')
                s.write(blob)
                if (i+1)%256==0 or i+1==rows:
                    print(f'  ESP32 computed {i+1}/{rows} pretrained INT8 rows',flush=True)
            line=s.readline().decode('ascii','replace').strip()
            if line!=f'RESULTS {rows*4}':raise RuntimeError(f'Expected RESULTS, got {line}')
            s.write(b'SEND\n')
            parts=[]
            remaining=rows*4
            deadline=time.monotonic()+30
            while remaining:
                chunk=s.read(remaining)
                if not chunk:
                    if time.monotonic()>deadline:raise RuntimeError('Truncated device results')
                    continue
                parts.append(chunk)
                remaining-=len(chunk)
            device_blob=b''.join(parts)
            device=np.frombuffer(device_blob,dtype='<f4').astype(np.float64)
            s.write(b'GOT\n')
            line=s.readline().decode('ascii','replace').strip()
            if not line.startswith('DONE P81 '):raise RuntimeError(f'Expected DONE: {line}')
            print(line,flush=True)
        reference=reference_for_rows(mm,m,x)
    if not np.all(np.isfinite(device)):raise RuntimeError('Nonfinite ESP32 output')
    delta=np.abs(device-reference)
    ok=np.isclose(device,reference,atol=args.atol,rtol=args.rtol)
    bad=int(np.count_nonzero(~ok))
    print('PHASE81 CHECK max_abs=',float(delta.max()),'mean_abs=',float(delta.mean()),'failures=',bad,'/',rows,flush=True)
    report={'layer':args.layer,'projection':args.projection,'rows':rows,'cols':cols,
            'max_abs':float(delta.max()),'mean_abs':float(delta.mean()),'failures':bad,
            'atol':args.atol,'rtol':args.rtol,'pass':bad==0}
    Path(args.report).write_text(json.dumps(report,indent=2),encoding='utf-8')
    if bad:raise SystemExit('PHASE81 DEVICE INT8 AUDIT FAIL')
    print('PHASE81 DEVICE INT8 AUDIT PASS',flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',default=str(DEFAULT_MANIFEST))
    p.add_argument('--port',default='COM3')
    p.add_argument('--layer',type=int,default=1)
    p.add_argument('--projection',choices=('q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'),default='q_proj')
    p.add_argument('--seed',type=int,default=81)
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--atol',type=float,default=0.002)
    p.add_argument('--rtol',type=float,default=0.0005)
    p.add_argument('--report',default='phase81_device_int8_report.json')
    run(p.parse_args())

if __name__=='__main__':main()
