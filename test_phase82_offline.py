import ast
import struct
from pathlib import Path
import numpy as np
p=Path(__file__).parent
src=(p/'main'/'esp32_c3_llm.c').read_text()
assert 'MATRIX %d %s %u %u' in src
assert 'HEAD_ROWS 151936 1792' in src
assert 'for(unsigned li=0;li<count;li++)' in src
for file in ('phase82_fullstack_w8.py',): ast.parse((p/file).read_text())
from phase82_fullstack_w8 import rms
x=np.ones(896,dtype=np.float32)
a=rms(x,np.ones_like(x))
assert np.max(np.abs(a-1))<1e-5
print('PHASE82 OFFLINE SYNTHETIC TEST PASS')
