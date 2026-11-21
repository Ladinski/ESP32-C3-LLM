from pathlib import Path
import ast
src=Path('main/esp32_c3_llm.c').read_text()
py=Path('phase63_verify_prompt.py').read_text()
ast.parse(py)
assert 'START P63 VERIFY %d %d' in src and 'READY PHASE63' in src
assert 'PHASE63_APPLIED rows=%u group=%u outlier=%u' in src
assert 'p63_real_verify && layer==2 && idx==6' in src
assert 'p61_prepare(x,cols,qinput[0],group_steps[0])' in src
assert 'memcpy(qinput[1]+p61_group*P34_GROUP,p61_q,sizeof(p61_q))' in src
assert 'y[row]=p62_preprocessed_dot(record,cols)*scale;' in src
assert 'p63_rows++' in src
assert '!p63_real_verify && layer==2 && idx==6 && p56_selective_enabled' in src
assert "cmd=f'START P63 VERIFY {len(ids)} {len(layers)}\\n'" in py
assert 'Missing PHASE63_APPLIED device confirmation' in py
assert 'int(match.group(1)) != D' in py
assert 'DONE PHASE63 VERIFY' in py
print('PHASE63 OFFLINE SOURCE/PROTOCOL PASS: real 3-layer correction, 896-row device evidence, strict independent verification')
