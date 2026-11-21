import numpy as np
from phase67_combined_mlp_gate import quant,plan,project,swiglu,metrics
rng=np.random.default_rng(67)
weights=rng.integers(-8,8,size=(5,256),dtype=np.int8)
scales=np.ones(5,np.float32)
x=rng.normal(size=256).astype(np.float32)
x[143]=1020.5
q,step=quant(x[128:256]); assert q.dtype==np.int16 and np.max(np.abs(q))<=32760
p=plan(x,1);assert p['group']==1 and p['indices'].tolist()==[15]
y0,_=project(weights,scales,x,0)
y1,p=project(weights,scales,x,1)
ref=weights.astype(np.float64)@x.astype(np.float64)
assert np.max(np.abs(y1-ref)) < np.max(np.abs(y0-ref))
assert np.all(np.isfinite(swiglu(np.array([-.2,3.],np.float32),np.array([1.,2.],np.float32))))
assert metrics(y1,y1)['failures']==0
print('PHASE67 OFFLINE TEST PASS: signed INT4 projection, Q15 scale, per-group peel, combined-stage arithmetic')
