from pathlib import Path
import re
src = Path('main/esp32_c3_llm.c').read_text()
runner = Path('bench_phase61.py').read_text()
assert 'START P61 BENCH' in src and 'READY PHASE61 BENCH' in src
assert 'PHASE61_KERNEL' in src and 'PHASE61_KERNEL' in runner
assert 'p61_peel_dot' in src and 'p61_prepare' in src
assert 'p61_outlier_value' in src and 'p61_weight_at' in src
assert 'return sum;' in src and 'dot_int4_grouped_q15' in src
assert 'free(w);free(x)' in src
assert 'ct[2]<=bt[2]' in src
assert 'selected=(BENCH_PASS|FALLBACK)' in runner
print('PHASE61 OFFLINE SOURCE TEST PASS: command, sparse outlier, row bench, strict speed gate')
