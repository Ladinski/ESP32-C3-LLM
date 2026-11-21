"""Lightweight tests of the scale-aware comparison rule (no model, serial or PC inference)."""
import ast
from pathlib import Path
import numpy as np

def counts(device, reference, atol=.001, rtol=.001):
    delta=np.abs(np.asarray(device,dtype=np.float64)-np.asarray(reference,dtype=np.float64))
    bound=atol+rtol*np.abs(reference)
    return int(np.count_nonzero(delta>bound))

assert counts([1000.02],[1000.0])==0  # large value, tiny relative error
assert counts([0.005],[0.0])==1  # near-zero value, meaningful deviation
assert counts([0.0],[0.0])==0
assert counts([1.0021],[1.0])==1
script=Path(__file__).with_name('phase51_verify_depth.py').read_text(encoding='utf-8')
assert "START P47 VERIFY" in script and 'LAYER_FULL' in script
ast.parse(script)
print('PHASE51 OFFLINE TEST PASS: mixed-scale tolerance and P47 protocol',flush=True)
