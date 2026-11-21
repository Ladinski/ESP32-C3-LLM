from phase74_layer_fidelity import measures
import numpy as np
x=np.array([1.0,2.0,3.0],dtype=np.float32)
y=x.copy()
r=measures(x,y,.001,.001)
assert r['failures']==0 and r['max_abs']==0 and abs(r['cosine']-1)<1e-10
z=np.array([1.0,2.05,3.0],dtype=np.float32)
assert measures(x,z,.001,.001)['failures']==1
print('PHASE74 OFFLINE TEST PASS: vector comparison, scale-aware tolerance and layer failure detection')
