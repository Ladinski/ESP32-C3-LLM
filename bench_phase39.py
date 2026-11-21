"""Benchmark-only Phase 39 serial test. No model transfer or inference required."""
import argparse
import time
import serial


def main():
    p=argparse.ArgumentParser(description='ESP32-C3 INT4 microbenchmark only')
    p.add_argument('--port',default='COM3')
    p.add_argument('--baud',type=int,default=115200)
    p.add_argument('--timeout',type=float,default=15)
    a=p.parse_args()
    deadline=time.monotonic()+a.timeout
    ready=False
    p38=None
    p39=None
    with serial.Serial(a.port,a.baud,timeout=.25,write_timeout=3) as dev:
        time.sleep(.6)
        dev.write(b'START KERNEL_BENCH\n')
        last_sent=time.monotonic()
        while time.monotonic()<deadline:
            raw=dev.readline()
            if not raw:
                if not ready and time.monotonic()-last_sent>1.5:
                    dev.write(b'START KERNEL_BENCH\n')
                    last_sent=time.monotonic()
                continue
            line=raw.decode('ascii',errors='replace').strip()
            if line.startswith(('READY PHASE39 BENCH','PHASE38_KERNEL','PHASE39_KERNEL','DONE PHASE39 BENCH','ERR ')):
                print(line,flush=True)
            if line=='READY PHASE39 BENCH': ready=True
            elif line.startswith('PHASE38_KERNEL '):p38=line
            elif line.startswith('PHASE39_KERNEL '):p39=line
            elif line.startswith('ERR '):raise RuntimeError(line)
            elif line=='DONE PHASE39 BENCH':break
        else:raise TimeoutError('No Phase 39 benchmark result; verify firmware was flashed, COM port, serial monitor closed')
    if not ready or p38 is None or p39 is None:raise RuntimeError('Incomplete benchmark output')
    parts={t.split('=',1)[0]:t.split('=',1)[1] for t in p39.split()[1:]}
    if parts.get('exact')!='1':raise RuntimeError('Exact INT4 arithmetic mismatch. Do NOT use candidate.')
    old=int(parts['baseline_us']); new=int(parts['candidate_us'])
    print(f'INT4 arithmetic only: baseline={old/1000:.3f}ms candidate={new/1000:.3f}ms',flush=True)
    if parts['selected']=='UNROLL16':
        print('PHASE 39 MICROBENCH PASS: candidate selected; proceed to two-layer inference verification.',flush=True)
    else:
        print('PHASE 39 MICROBENCH PASS: candidate rejected; keep Phase 38 kernel. Skip the inference run.',flush=True)

if __name__=='__main__':main()
