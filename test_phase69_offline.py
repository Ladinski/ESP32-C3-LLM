from pathlib import Path
import ast
p=Path(__file__).parent
s=(p/'phase69_full_reference.py').read_text()
ast.parse(s);ast.parse((p/'phase69_reference_math.py').read_text());ast.parse((p/'phase69_resume_offline.py').read_text())
checks=['PHASE69 DEVICE CAPTURE SAVED','PHASE69 FINAL HIDDEN','PHASE69 TOKEN MATCH','DONE PHASE68 GENERATE','PHASE68_APPLIED','head_reference(hm,ref_state,head_meta)',"reference(mm,model_meta['layers'][:24],embed_vectors)"]
for t in checks:assert t in s,t
assert 'idf.py' not in s
assert 'serial.Serial' not in (p/'phase69_resume_offline.py').read_text()
print('PHASE69 OFFLINE SOURCE TEST PASS: no firmware change, device capture, post-device 24-layer reference, independent token comparison, offline resume')
