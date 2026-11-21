"""Lightweight source/protocol regression checks. No pretrained weights or ESP32 needed."""
from pathlib import Path
import ast
from phase70_two_token_verify import D
assert D == 896
src=Path(__file__).with_name('phase70_two_token_verify.py').read_text()
ast.parse(src)
ast.parse(Path(__file__).with_name('phase70_resume_offline.py').read_text())
assert 'len(ids)!=2' in src
assert "'START P68 '+mode" in src
assert 'requested_positions!=[0,1]' in src
assert 'completed_positions!=[0,1]' in src
assert 'layer_progress={0:set(),1:set()}' in src
assert 'PHASE70 DEVICE CAUSAL PATH PASS' in src
assert 'reference_positions=np.stack([expected[pos,23] for pos in (0,1)])' in src
assert 'PHASE70 TOKEN MATCH' in src
print('PHASE70 OFFLINE SOURCE/PROTOCOL PASS: strict two-token, 24-layer causal audit, per-position capture, resumable reference')
