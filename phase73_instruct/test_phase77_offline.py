import numpy as np
from phase77_fp32_math_gate import rms,silu,quality
x=np.array([1.,-2.,3.],dtype=np.float32)
assert np.allclose(rms(x,np.ones(3,dtype=np.float32),1e-6),x/np.sqrt(np.mean(x*x)+1e-6))
assert np.allclose(silu(x),x/(1+np.exp(-x)))
assert quality(x,x,.001,.001)['fails']==0
assert quality(x,x+0.1,.001,.001)['fails']==3
print('PHASE77 SYNTHETIC TEST PASS')
