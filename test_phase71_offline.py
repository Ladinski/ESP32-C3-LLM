"""Static regression test; does not claim a real ESP32 build or performance."""
from pathlib import Path
import ast
src=Path('phase71_generate.py').read_text()
fw=Path('main/esp32_c3_llm.c').read_text()
for f in ['phase71_generate.py','phase71_support.py']:
    ast.parse(Path(f).read_text())
assert 'START P71 GENERATE 2 24' in fw and "START P71 GENERATE 2 24" in src
assert 'READY PHASE71' in fw and 'READY PHASE71' in src
assert 'PHASE71_STEP %d %d' in fw and 'PHASE71_STEP ' in src
assert 'PHASE71_APPLIED rows=%u' in fw
assert 'p46_token_count=2' in fw
assert 'if(!phase45_output_head())return false;' in fw
assert 'p71_last_selected=best_id;' in fw
assert 'kv_keys[P25_LAYERS][2][KV]' in fw
assert 'PHASE71 PASS' in src
assert 'reference_id!=token' in src
assert "id_to_embed=chosen[0][0]" in src
assert 'p71_generate=true' in fw
print('PHASE71 OFFLINE SOURCE/PROTOCOL TEST PASS: 2 selected heads, causal feedback, 2-slot KV, no host token choice')
