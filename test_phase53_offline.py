import numpy as np
from phase53_downprojection_audit import grouped_q15,metrics
x=np.zeros(4864,dtype=np.float32)
x[:3]=[0.25,-3.0,7.2]
q,s=grouped_q15(x)
assert np.allclose((q[:128].astype(np.float32)*s[0])[:3],x[:3],rtol=2e-4,atol=0.0002)
assert (q[128:]==0).all()
assert metrics(np.array([0.0,0.02],dtype=np.float32))['count_above_0_001']==1
print('PHASE53 OFFLINE TEST PASS: grouped Q15 replay, zero groups, threshold metrics')
