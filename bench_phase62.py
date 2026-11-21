"""Phase 62 one-outlier prequantization benchmark, ESP32-C3 USB Type-C."""
import argparse
import re
import serial
import sys
import time

PATTERN=re.compile(r"PHASE62_KERNEL valid=(\d+) group=(\d+) outlier=(\d+) baseline_us=(\d+) candidate_us=(\d+) p61_us=(\d+) base_error=([0-9.eE+-]+) candidate_error=([0-9.eE+-]+) selected=(BENCH_PASS|FALLBACK)")

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--port',default='COM3')
    ap.add_argument('--baud',type=int,default=115200)
    ap.add_argument('--timeout',type=float,default=50)
    args=ap.parse_args()
    with serial.Serial(args.port,args.baud,timeout=0.5,write_timeout=3) as dev:
        time.sleep(3)
        dev.reset_input_buffer()
        dev.write(b'START P62 BENCH\n')
        dev.flush()
        print("SENT 'START P62 BENCH'",flush=True)
        deadline=time.monotonic()+args.timeout
        ready=False; done=False; result=None
        while time.monotonic()<deadline:
            line=dev.readline().decode('utf-8','replace').strip()
            if not line: continue
            print(line,flush=True)
            ready|=line=='READY PHASE62 BENCH'
            done|=line=='DONE PHASE62 BENCH'
            if m:=PATTERN.fullmatch(line):result=m
            if done:break
        if not (ready and done and result):
            raise RuntimeError('Phase 62 protocol incomplete; do not start inference')
        valid,g,outlier,b,c,p,errb,errc,selected=result.groups()
        b=int(b);c=int(c);p=int(p)
        if b<=0 or c<=0 or p<=0:raise RuntimeError('Zero benchmark timings')
        if selected=='BENCH_PASS' and (valid!='1' or c>b or float(errc)>float(errb)):
            raise RuntimeError('Hardware gate incorrectly marked BENCH_PASS')
        print(f'ROW KERNEL: baseline={b/1000:.3f} ms, phase61={p/1000:.3f} ms, phase62={c/1000:.3f} ms; candidate change={100*(c/b-1):+.2f}%')
        if selected=='BENCH_PASS':
            print('PHASE62 BENCH PASS: synthetic speed + accuracy only. Pretrained-model inference still NOT verified.')
        else:
            print('PHASE62 FALLBACK: retain the original SHIFT + grouped Q15. Do NOT run full-model inference.')

if __name__=='__main__':
    try:main()
    except Exception as e:
        print(f'PHASE62 ERROR: {e}',file=sys.stderr)
        sys.exit(1)
