from pathlib import Path
import ast
import re
src=Path('main/esp32_c3_llm.c').read_text()
runner=Path('bench_phase62.py').read_text()
ast.parse(runner)
assert 'START P62 BENCH' in src and 'READY PHASE62 BENCH' in src and 'DONE PHASE62 BENCH' in src
assert 'p62_preprocessed_dot' in src and 'p62_benchmark' in src
body=src.split('static float p62_preprocessed_dot(',1)[1].split('static void p62_benchmark(',1)[0]
assert 'dot_int4_grouped_q15(packed,qinput[1],cols,group_steps[1])' in body
assert 'p61_weight_at(special,p61_outlier_local)*p61_outlier_value' in body
assert 'p61_peel_dot(' not in body
assert 'memcpy(qinput[1],qinput[0],sizeof(qinput[0]));' in src
assert 'memcpy(group_steps[1],group_steps[0],sizeof(group_steps[0]));' in src
assert 'for(uint32_t test=0;test<16U;test++)' in src
assert 'ct[3]<=bt[3]' in src
assert 'free(w);free(x)' in src
assert 'selected=(BENCH_PASS|FALLBACK)' in runner
print('PHASE62 OFFLINE SOURCE TEST PASS: original kernel reused, one scalar correction, 16-row equivalence, 7-pass speed gate')
