from pathlib import Path
p=Path('main/esp32_c3_llm.c').read_text(encoding='utf-8')
r=Path('phase68_full_stack_token.py').read_text(encoding='utf-8')
for text in ['START P68 GENERATE %d %d','READY PHASE68','PHASE68_APPLIED','DONE PHASE68 GENERATE','p63_real_verify=true;p68_generate=true','p46_token_count*D']:
    assert text in p,text
for text in ['START P68 ','READY PHASE68','PHASE68_APPLIED','DONE PHASE68 ','head_reference','--active-layers','a.active_layers!=24']:
    assert text in r,text
assert 'p63_correct_this=(p63_real_verify && layer==2 && idx==6' in p
assert 'phase45_output_head' in p
print('PHASE68 SOURCE/PROTOCOL TEST PASS: 24-layer prompt, correction proof, real on-device vocabulary head')
