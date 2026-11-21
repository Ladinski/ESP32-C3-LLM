"""Regression for actual Phase68 firmware order: TOKEN COMPLETE precedes OUT chunks."""
import ast
from pathlib import Path
s=Path(__file__).with_name('phase70_two_token_verify.py').read_text(encoding='utf-8')
ast.parse(s)
assert 'Firmware emits TOKEN <pos> COMPLETE BEFORE streaming OUT chunks.' in s
assert 'len(layer_progress[pos])!=24' in s
assert 'len(state_chunks[pos])!=D//16 or len(layer_progress[pos])!=24' not in s
assert "if pos>0 and (completed_positions!=list(range(pos)) or len(state_chunks[pos-1])!=D//16):" in s
assert "if pos not in completed_positions:raise RuntimeError('OUT chunk before TOKEN COMPLETE: '+line)" in s
# Simulate the actual serial ordering with the same invariants.
D=896
completed=[]
chunks={0:set(),1:set()}
layers={0:set(range(24)),1:set(range(24))}
for pos in range(2):
    assert pos==len(completed) and len(layers[pos])==24
    completed.append(pos)  # firmware's TOKEN pos COMPLETE
    for start in range(0,D,16):
        assert pos in completed and start not in chunks[pos]
        chunks[pos].add(start)  # firmware's subsequent OUT pos start lines
    if pos==0:assert completed==[0] and len(chunks[0])==D//16
assert completed==[0,1] and all(len(chunks[i])==D//16 for i in range(2))
print('PHASE70 ORDER FIX TEST PASS: TOKEN COMPLETE then OUT, both positions, 24 layers and 896 channels')
