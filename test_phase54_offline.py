import numpy as np
from phase54_group_precision_gate import summarize
from phase53_downprojection_audit import grouped_q15

# Exact groups must represent identical quantized activations, and group counts scale.
x=np.zeros(4864,np.float32)
x[127]=50.0
x[128]=0.001
x[813]=-1359.87
x[2661]=15.13
for size in (128,64,32,16):
    q,scales=grouped_q15(x,size)
    assert q.shape==(4864,) and scales.shape==(4864//size,)
    assert np.isfinite(scales).all()
    assert abs(float((q.astype(np.float32)*np.repeat(scales,size))[813])-x[813])<0.05

ref=np.array([1.,2.,3.],np.float32)
exact=np.array([1.0001,2.,3.],np.float32)
preds={128:np.array([1.1,2.1,3.],np.float32),64:np.array([1.,2.,3.],np.float32),32:ref.copy(),16:ref.copy()}
r=summarize(ref,exact,preds)
assert r['candidates'][0]['group_size']==128
assert r['baseline_128_failures']==2
assert r['candidates'][1]['offline_precision_gate']
print('PHASE54 OFFLINE TEST PASS: groups 128/64/32/16, error attribution and candidate gate')
