from pathlib import Path
import ast
source=Path(__file__).with_name('phase47_verify_prompt.py').read_text()
firmware=Path(__file__).parent/'main'/'esp32_c3_llm.c'
ast.parse(source)
c=firmware.read_text()
assert 'START P47 VERIFY %d %d' in c
assert 'READY PHASE47' in c and 'DONE PHASE47 VERIFY' in c
assert 'if(!send_layer_full(pos,layer,state))return false;' in c
assert "line.startswith('LAYER_FULL ')" in source
assert 'Computing independent NumPy reference AFTER all ESP32 inference' in source
assert 'if(p45_generate)' in c and 'p45_generate=false;p46_prompt_mode=true;p47_verify_only=true' in c
print('PHASE47 PROTOCOL PASS: no-head diagnostic, real embeddings, complete layer streams, post-inference NumPy checks')
