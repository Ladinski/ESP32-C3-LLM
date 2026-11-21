"""Offline Phase 34 check: pretrained weights stay signed INT4; Q15 group math.
Uses Python standard-library randomness to avoid NumPy random DLL restrictions.
"""
import random
import numpy as np


def quantize(x):
    maxabs = float(np.max(np.abs(x)))
    if maxabs == 0:
        return np.zeros(len(x),dtype=np.int16),1.0
    q = np.clip(np.rint(x*(32760.0/maxabs)),-32760,32760).astype(np.int16)
    return q,maxabs/32760.0


def test():
    rng=random.Random(34)
    improvements=[]
    for dims in (896,4864):
        for trial in range(32):
            x=np.fromiter((rng.gauss(0.0,1.0) for _ in range(dims)), dtype=np.float32, count=dims)
            if trial%4==0:x[trial]=30.0 # outlier tests
            w=np.fromiter((rng.randrange(-8,8) for _ in range(dims)), dtype=np.int16, count=dims) # signed INT4 unchanged
            packed=(w[0::2]&15)|((w[1::2]&15)<<4)
            restored=np.empty_like(w)
            lo=(packed&15).astype(np.int16);hi=(packed>>4).astype(np.int16)
            restored[0::2]=np.where(lo>=8,lo-16,lo)
            restored[1::2]=np.where(hi>=8,hi-16,hi)
            assert np.array_equal(w,restored), 'INT4 bytes were changed'
            original=float(np.dot(w.astype(np.float64),x.astype(np.float64)))
            q,s=quantize(x)
            global_sum=float(np.dot(w.astype(np.int64),q.astype(np.int64))*s)
            group_sum=0.0
            for start in range(0,dims,128):
                sub,scale=quantize(x[start:start+128])
                group_sum+=float(np.dot(w[start:start+128].astype(np.int64),sub.astype(np.int64))*scale)
            assert np.isfinite(group_sum)
            improvements.append((abs(global_sum-original),abs(group_sum-original)))
    before=np.array(improvements)
    print(f'PHASE 34 INT4 GROUP QUANTIZATION PASS | cases={len(before)}')
    print(f'Average abs error: vector-Q15={before[:,0].mean():.7g} grouped-Q15={before[:,1].mean():.7g}')
    print(f'Worst abs error:   vector-Q15={before[:,0].max():.7g} grouped-Q15={before[:,1].max():.7g}')
    print('This is an offline arithmetic fixture, not pretrained-layer or hardware verification.')

if __name__=='__main__':test()
