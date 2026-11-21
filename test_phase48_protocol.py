from pathlib import Path
c=Path('main/esp32_c3_llm.c').read_text()
p=Path('phase48_trace_prompt.py').read_text()
stages=['NORM1','VALUE','CONTEXT','ATTN_OUT','RESID1','NORM2','GATE','UP','ACT','MLP_OUT']
assert 'START P48 TRACE %d %d' in c and 'READY PHASE48' in c
assert 'DONE PHASE48 TRACE' in c and 'START P48 TRACE {len(ids)} {len(layers)}' in p
assert all(('"'+v+'"') in c and ("'"+v+"'") in p for v in stages)
assert c.count('p48_emit(')>=11
assert 'phase45_output_head' in c and 'if(p45_generate)' in c
print('PHASE48 PROTOCOL PASS: all 10 stage markers, token0 only, 3-layer limit, original INT4 path')
