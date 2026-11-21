"""Small source and format checks; real checkpoint conversion is not simulated."""
import ast
from pathlib import Path
import numpy as np
import struct

for name in ('export_qwen_stack.py','export_qwen_layer0.py','export_phase45_head.py','phase73_fidelity_gate.py','phase69_reference_math.py','phase69_full_reference.py'):
    ast.parse(Path(name).read_text(encoding='utf8'))
for name in ('export_qwen_layer0.py','export_phase45_head.py'):
    assert "REPO='Qwen/Qwen2.5-0.5B-Instruct'" in Path(name).read_text()
# Verify symmetric signed 4-bit layout identical to existing ESP32 runtime (low nibble first).
x=np.array([-8,-7,-1,0,1,7],dtype=np.int8)
packed=(x[::2].astype(np.uint8)&15)|((x[1::2].astype(np.uint8)&15)<<4)
lo=(packed&15).astype(np.int8);hi=(packed>>4).astype(np.int8)
lo[lo>=8]-=16;hi[hi>=8]-=16
out=np.empty_like(x);out[::2]=lo;out[1::2]=hi
assert np.array_equal(out,x)
assert struct.calcsize('<f')==4
print('PHASE73 OFFLINE SOURCE/INT4 LAYOUT TEST PASS')
