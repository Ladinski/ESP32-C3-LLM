"""ESP32-C3 Phase 58 one-group dual-Q15 residual kernel speed gate.

Uses USB-C COM3 for a short synthetic arithmetic test only. Does not stream
model data or perform inference. If candidate loses, leave inference unchanged.
"""
import argparse
import re
import sys
import time
import serial

RESULT = re.compile(r'^PHASE58_KERNEL valid=(\d+) group=(\d+) baseline_us=(\d+) candidate_us=(\d+) base_error=([0-9eE.+-]+) candidate_error=([0-9eE.+-]+) selected=(BENCH_PASS|FALLBACK)$')


def run(port, baud, timeout):
    with serial.Serial(port, baudrate=baud, timeout=.5, write_timeout=3) as dev:
        time.sleep(3)
        dev.reset_input_buffer()
        dev.write(b'START P58 BENCH\n')
        dev.flush()
        print("SENT 'START P58 BENCH'", flush=True)
        deadline=time.monotonic()+timeout
        ready=done=False
        match=None
        while time.monotonic()<deadline:
            line=dev.readline().decode('utf-8','replace').strip()
            if not line:continue
            print(line,flush=True)
            if line=='READY PHASE58 BENCH':ready=True
            found=RESULT.fullmatch(line)
            if found: match=found
            if line=='DONE PHASE58 BENCH':done=True;break
        if not (ready and done and match):
            raise RuntimeError('Phase 58 protocol incomplete. Verify flashed source; do not run inference.')
        valid, group, old_us, new_us, old_err, new_err, choice=match.groups()
        old_us,new_us=int(old_us),int(new_us)
        if choice=='BENCH_PASS' and (valid!='1' or new_us>old_us):
            raise RuntimeError('Firmware reported a pass that violates the speed/accuracy gate')
        print(f'ROW KERNEL: baseline={old_us/1000:.3f} ms, candidate={new_us/1000:.3f} ms, change={100*(new_us/old_us-1):+.2f}%',flush=True)
        if choice=='FALLBACK':
            print('PHASE58 REJECT: keep original SHIFT + grouped Q15. No inference test.',flush=True)
        else:
            print('PHASE58 MICROBENCH PASS: do not treat as full inference verification. Report this result before enabling candidate.',flush=True)
        return choice


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--port',default='COM3')
    ap.add_argument('--baud',type=int,default=115200)
    ap.add_argument('--timeout',type=float,default=30)
    a=ap.parse_args()
    run(a.port,a.baud,a.timeout)

if __name__=='__main__':
    try:main()
    except Exception as e:
        print('PHASE58 ERROR:',e,file=sys.stderr)
        sys.exit(1)
