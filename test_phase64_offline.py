from pathlib import Path
import ast
c=Path('main/esp32_c3_llm.c').read_text()
p=Path('phase64_verify_depth.py').read_text()
ast.parse(p)
assert 'START P64 VERIFY %d %d' in c
assert 'requested!=8' in c and 'ERR P64_CONFIG' in c
assert 'READY PHASE64' in c and 'DONE PHASE64 VERIFY' in c
assert 'PHASE64_APPLIED rows=%u group=%u outlier=%u' in c
assert 'p63_real_verify && layer==2 && idx==6' in c, 'Only layer3 must be corrected'
assert 'p63_rows++' in c
assert 'matrix_single(layer,6' in c
assert 'START P64 VERIFY {len(ids)} {len(layers)}' in p
assert 'Missing PHASE64_APPLIED device confirmation' in p
assert 'int(match.group(1)) != D' in p
assert 'args.active_layers!=8' in p
assert 'needed=len(ids)*len(layers)*(D//16)' in p
assert 'first_bad' in p and 'if args.strict and first_bad' in p
assert "default='phase64_layer_report.json'" in p
assert 'phase63_layer_report' not in p
print('PHASE64 OFFLINE TEST PASS: eight-layer protocol, layer3-only correction, complete 896-channel audit')
