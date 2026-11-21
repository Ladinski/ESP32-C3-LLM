"""Offline source and runner regression checks; cannot prove hardware performance."""
from pathlib import Path
import ast
import re

here=Path(__file__).resolve().parent
src=(here/'main'/'esp32_c3_llm.c').read_text()
host=(here/'bench_phase56.py').read_text()
ast.parse(host)
assert 'START P56 BENCH' in src and 'READY PHASE56 BENCH' in src
assert 'p56_find_group' in src and 'p56_corrected_dot' in src
assert 'layer==2 && idx==6 && p56_selective_enabled' in src
assert 'ct[1]<=bt[1]' in src
assert 'phase38_select_kernel();' in src
assert 'P34_GROUP 128U' in src
assert "sum+=dot_fp32(packed+begin/2,x+begin,count);" in src
assert 'START P56 BENCH' in host and 'SELECTIVE|FALLBACK' in host
print('PHASE56 OFFLINE PROTOCOL PASS: dedicated bench, selective down projection, automatic fallback')
