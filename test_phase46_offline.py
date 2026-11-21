"""Phase 46 static protocol smoke tests; no pretrained model needed."""
from pathlib import Path
source=Path(__file__).parent/'main/esp32_c3_llm.c'
runner=Path(__file__).parent/'phase46_prompt_runner.py'
s=source.read_text();r=runner.read_text()
checks={
'bounded prompt': 'p46_token_count<1 || p46_token_count>2' in s,
'ESP32 embedding receive': 'phase45_read_exact(state,D*4U)' in s,
'ESP32 all layer computations': 'run_token_major_decode()' in s,
'ESP32 on-device head': 'phase45_output_head()' in s,
'host only embedding rows': "get_row('model.embed_tokens.weight',token)" in r,
'head independent reference AFTER hardware': 'Computing independent reference AFTER ESP32 finished' in r,
'full stack limitation disclosed': 'does NOT validate the full transformer' in r,
'separate protocol': 'START P46 PROMPT' in s and "'START P46 '+mode" in r,
}
for k,v in checks.items():print(('PASS' if v else 'FAIL'),k)
if not all(checks.values()):raise SystemExit(1)
print('PHASE46 STATIC PROTOCOL PASS; actual ESP32 verification still required')
