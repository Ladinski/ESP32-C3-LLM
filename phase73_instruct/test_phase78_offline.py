import numpy as np
from phase78_int8_gate import quantize_int8_per_row_numpy,stats

rng=np.random.default_rng(42)
w=rng.normal(size=(17,36)).astype(np.float32)
q=quantize_int8_per_row_numpy(w)
assert q.shape==w.shape
assert np.max(np.abs(w-q)) <= np.max(np.abs(w),axis=1).max()/127 + 1e-6
assert np.all(quantize_int8_per_row_numpy(np.zeros((3,4),np.float32))==0)
r=stats(w.ravel(),q.ravel())
assert r['size']==17*36 and r['cosine']>0.99
try:
    import torch
    from phase78_int8_gate import quantize_int8_per_row_inplace
    t=torch.from_numpy(w.copy())
    quantize_int8_per_row_inplace(t)
    np.testing.assert_allclose(t.numpy(),q,atol=1e-6,rtol=1e-6)
except ImportError:
    pass
print('PHASE78 OFFLINE SYNTHETIC TEST PASS')
