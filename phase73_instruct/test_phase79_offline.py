"""Lightweight synthetic tests requiring no downloads."""
import ast
from pathlib import Path

root = Path(__file__).resolve().parent
src = root.joinpath('phase79_multitoken_gate.py').read_text(encoding='utf8')
ast.parse(src)
from phase79_multitoken_gate import compare
assert compare([1, 2, 3], [1, 2, 3])['exact_match']
assert compare([1, 2, 3], [1, 2, 4])['first_different_generated_position'] == 3
assert compare([1, 2], [1, 2, 4])['exact_match'] is False
assert compare([1, 2], [1, 2, 4])['matching_prefix_tokens'] == 2
print('PHASE79 OFFLINE SYNTHETIC TEST PASS')
