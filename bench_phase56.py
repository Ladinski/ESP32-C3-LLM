"""Phase 56 ESP32-C3 selective-correction microbenchmark. No model weights streamed."""
import argparse
import re
import sys
import time

import serial

LINE = re.compile(r"^PHASE56_KERNEL valid=(\d+) group=(\d+) baseline_us=(\d+) candidate_us=(\d+) base_error=([\d.eE+-]+) candidate_error=([\d.eE+-]+) selected=(SELECTIVE|FALLBACK)$")


def run(port: str, baud: int, timeout: float):
    with serial.Serial(port, baudrate=baud, timeout=0.5, write_timeout=3) as dev:
        time.sleep(3)
        dev.reset_input_buffer()
        command=b'START P56 BENCH\n'
        dev.write(command)
        dev.flush()
        print("SENT 'START P56 BENCH'", flush=True)
        deadline=time.monotonic()+timeout
        fields=None
        ready=False
        done=False
        while time.monotonic()<deadline:
            line=dev.readline().decode('utf-8','replace').strip()
            if not line:
                continue
            print(line,flush=True)
            if line=='READY PHASE56 BENCH':ready=True
            match=LINE.fullmatch(line)
            if match: fields=match
            if line=='DONE PHASE56 BENCH':done=True; break
        if not (ready and done and fields):
            raise RuntimeError('Phase 56 bench protocol incomplete. Confirm firmware and COM3; do NOT run inference.')
        valid, group, base_us, cand_us, base_err, cand_err, choice = fields.groups()
        base_us=int(base_us);cand_us=int(cand_us)
        if choice=='SELECTIVE' and (valid!='1' or cand_us>base_us):
            raise RuntimeError('Firmware selected a candidate that violated benchmark rules')
        print(f'KERNEL PERFORMANCE: {base_us/1000:.3f}ms baseline vs {cand_us/1000:.3f}ms candidate | {100*(cand_us/base_us-1):+.2f}% runtime',flush=True)
        if choice=='FALLBACK':
            print('PHASE56 REJECT: candidate not fast enough or numerical check failed. Original SHIFT + grouped Q15 retained. DO NOT run 3-layer accuracy.',flush=True)
        else:
            print(f'PHASE56 CANDIDATE ENABLED for this session: chosen activation group {group}. Next test 3-layer numerical accuracy.',flush=True)
        return choice


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--port',default='COM3')
    ap.add_argument('--baud',type=int,default=115200)
    ap.add_argument('--timeout',type=float,default=25)
    args=ap.parse_args()
    run(args.port,args.baud,args.timeout)

if __name__=='__main__':
    try: main()
    except Exception as exc:
        print(f'PHASE56 BENCH ERROR: {exc}',file=sys.stderr)
        sys.exit(1)
