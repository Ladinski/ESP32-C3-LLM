"""Phase 22: independent, deterministic Qwen-style block reference and serial check.
Only reference/testing happens on Python. Device uses own seeded weights + math.
This is NOT pretrained Qwen inference.
"""
import argparse
import time
import numpy as np

D, QH, KVH, HD, FF, STEPS = 32, 4, 2, 8, 64, 5


def matrix(rows, cols, seed):
    w = np.fromiter(((((r*17+c*13+seed*7+(r*c)%11)%15)-7)
                     for r in range(rows) for c in range(cols)),
                    dtype=np.float32, count=rows*cols).reshape(rows, cols)
    scales = np.array([.009+.001*((r+seed)%7) for r in range(rows)], dtype=np.float32)
    return w*scales[:, None]

WQ, WK, WV, WO, WG, WU, WD = (
    matrix(D,D,1),matrix(KVH*HD,D,2),matrix(KVH*HD,D,3),
    matrix(D,D,4),matrix(FF,D,5),matrix(FF,D,6),matrix(D,FF,7)
)
G1 = 1+.01*(np.arange(D)%9-4)
G2 = 1+.01*(np.arange(D)%7-3)


def rmsnorm(x, gamma):
    return x/np.sqrt(np.mean(x*x)+1e-6)*gamma


def rope(arr, heads, pos):
    result=arr.reshape(heads,HD).copy()
    for j in range(HD//2):
        theta=pos/(10000.0**(2*j/HD))
        c,s=np.cos(theta),np.sin(theta)
        a,b=result[:,2*j].copy(),result[:,2*j+1].copy()
        result[:,2*j]=a*c-b*s
        result[:,2*j+1]=a*s+b*c
    return result


def reference():
    keys=[]; values=[]; results=[]
    for pos in range(STEPS):
        x=.02*np.array([(pos*7+d*3)%19-9 for d in range(D)],dtype=np.float64)
        n=rmsnorm(x,G1)
        q=rope(WQ.astype(np.float64)@n,QH,pos)
        k=rope(WK.astype(np.float64)@n,KVH,pos)
        v=(WV.astype(np.float64)@n).reshape(KVH,HD)
        keys.append(k); values.append(v)
        context=np.zeros((QH,HD))
        for h in range(QH):
            kvh=h//(QH//KVH)
            score=np.array([q[h]@keys[t][kvh]/np.sqrt(HD) for t in range(pos+1)])
            p=np.exp(score-score.max()); p/=p.sum()
            context[h]=sum(p[t]*values[t][kvh] for t in range(pos+1))
        residual=x+(WO.astype(np.float64)@context.reshape(-1))
        n2=rmsnorm(residual,G2)
        gate=WG.astype(np.float64)@n2
        up=WU.astype(np.float64)@n2
        hidden=(gate/(1+np.exp(-gate)))*up
        out=residual+(WD.astype(np.float64)@hidden)
        results.append(out)
    return np.array(results)


def check_rows(rows, threshold=3e-5):
    expected=reference()
    if sorted(rows)!=list(range(STEPS)):
        raise AssertionError(f'Expected output rows 0..{STEPS-1}, saw {sorted(rows)}')
    maximum=0.0
    for pos in range(STEPS):
        got=np.asarray(rows[pos], dtype=np.float64)
        if got.shape!=(D,) or not np.isfinite(got).all():
            raise AssertionError(f'Invalid output row {pos}: shape={got.shape}')
        err=float(np.max(np.abs(got-expected[pos])))
        maximum=max(maximum,err)
        print(f'  Token position {pos}: maximum difference = {err:.9g}')
        if err>threshold:
            raise AssertionError(f'Position {pos} numerical mismatch {err} > {threshold}')
    print(f'PHASE 22 PASS: RMSNorm + RoPE + GQA + SwiGLU + per-row INT4 matvec + KV cache verified; max error {maximum:.9g}')


def run_serial(port_name, baud, timeout):
    import serial
    rows={}; cache=[]; banner=False; done=False
    deadline=time.monotonic()+timeout
    with serial.Serial(port_name, baudrate=baud, timeout=.5) as dev:
        # Serial open resets ESP32-C3 USB JTAG on many boards.
        while time.monotonic()<deadline:
            b=dev.readline()
            if not b: continue
            line=b.decode('utf-8',errors='replace').strip()
            if not line: continue
            if line.startswith(('ESP32-C3 PHASE 22','CONFIG','MEM','OUT','CACHE','DONE')):
                print(line if not line.startswith('OUT ') else line[:70]+' ...')
            if line.startswith('ESP32-C3 PHASE 22'): banner=True
            if line.startswith('OUT '):
                parts=line.split(); pos=int(parts[1]); rows[pos]=[float(x) for x in parts[2:]]
            if line.startswith('CACHE '): cache.append(line)
            if line.startswith('DONE PHASE22'): done=True; break
    if not banner: raise RuntimeError('No Phase 22 banner. Verify flash, COM port and close idf.py monitor.')
    if not done: raise TimeoutError(f'No DONE PHASE22 within {timeout}s. Received rows: {sorted(rows)}')
    if len(cache)!=STEPS: raise AssertionError(f'Expected {STEPS} CACHE reports; received {len(cache)}')
    check_rows(rows)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--self-test',action='store_true',help='Run independent Python reference and invariants (no ESP32).')
    p.add_argument('--port',default='COM3')
    p.add_argument('--baud',type=int,default=115200)
    p.add_argument('--timeout',type=float,default=60)
    a=p.parse_args()
    if a.self_test:
        y=reference()
        assert y.shape==(STEPS,D) and np.isfinite(y).all()
        assert not np.allclose(y[0],y[-1])
        assert (WQ.min()<0) and (WQ.max()>0)
        print('PHASE 22 PC REFERENCE PASS | five autoregressive positions, 32 features')
        print('First token first eight:',y[0,:8].tolist())
        print('Last token first eight:',y[-1,:8].tolist())
    else: run_serial(a.port,a.baud,a.timeout)

if __name__=='__main__':main()
