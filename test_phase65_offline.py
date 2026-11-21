from pathlib import Path
import ast
c=Path('main/esp32_c3_llm.c').read_text()
p=Path('phase65_trace_layer4.py').read_text()
ast.parse(p)
assert 'START P65 TRACE %d %d' in c and 'requested!=4' in c
assert 'READY PHASE65' in c and 'DONE PHASE65 TRACE' in c
assert 'p65_trace ? layer!=3 : layer!=2' in c
assert 'PHASE65_APPLIED rows=%u group=%u outlier=%u' in c
assert 'p63_real_verify && layer==2 && idx==6' in c
assert 'START P65 TRACE {len(ids)} {len(layers)}' in p
assert 'P48_STAGE' in p and 'stage_received' in p
assert 'PHASE65 FIRST FAILING STAGE:' in p
assert 'layer4_stages=stage_results' in p
assert 'args.active_layers!=4' in p
print('PHASE65 OFFLINE SOURCE TEST PASS: layer4 stages, four pretrained layers, layer3 correction maintained')
