import ast
from pathlib import Path
import numpy as np
from phase66_attribution import quant,metrics
src=Path('phase66_capture_layer4.py').read_text()
ast.parse(src)
assert 'np.savez_compressed(args.trace_output,**saved)' in src
assert "'device_'+name" in src and "'reference_'+name" in src
v=np.array([2500,0.021,-.07,0.15,0.08],np.float32)
q,s=quant(v);assert np.max(np.abs(q.astype(np.float64)*s-v))<.05
assert metrics(np.array([1.],np.float32),np.array([1.],np.float32))['failures']==0
print('PHASE66 OFFLINE TEST PASS: trace capture, q15 and metrics')
