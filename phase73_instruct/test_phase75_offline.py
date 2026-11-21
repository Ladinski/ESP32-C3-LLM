import numpy as np
from phase75_projection_attribution import unpack_rows, int4_matvec, vector_metrics
from struct import pack
class ByteMmap:
    def __init__(self, b):self.b=b
    def __getitem__(self, k):return self.b[k]
q=np.array([[7,-8,3,0],[-1,2,-3,4]],dtype=np.int8)
scales=[.25,1.5]
raw=b''.join(bytes((int(q[i,0])&15)|((int(q[i,1])&15)<<4) for i in []) for _ in []) if False else b''
for i in range(2):
    raw+=bytes([(int(q[i,0])&15)|((int(q[i,1])&15)<<4),(int(q[i,2])&15)|((int(q[i,3])&15)<<4)])+pack('<f',scales[i])
w=unpack_rows(raw,2,4)
assert np.allclose(w,q*np.array(scales)[:,None])
x=np.array([1.,2.,3.,4.],dtype=np.float32)
m={'offset':0,'rows':2,'cols':4,'row_bytes':6}
y,weight_err=int4_matvec(ByteMmap(raw),m,x,rows_per_block=1,original_weight=w)
assert np.allclose(y,w@x)
assert weight_err['mae']==0
assert vector_metrics(y,y)['failures']==0
assert vector_metrics(y,y+.5)['failures']==2
print('PHASE75 OFFLINE TEST PASS: INT4 decode, signed weights, row offsets, projection arithmetic and metrics')
