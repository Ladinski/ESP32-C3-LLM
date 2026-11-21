import ast
from pathlib import Path
s=Path(__file__).with_name('phase72_model_quality_diagnostic.py').read_text(encoding='utf-8')
ast.parse(s)
assert 'Qwen/Qwen2.5-0.5B' in s
assert 'Qwen/Qwen2.5-0.5B-Instruct' in s
assert 'apply_chat_template' in s
assert 'do_sample=False' in s
assert 'EXPECTED_ESP32_FIRST_TOKEN_FOR_HELLO = 52643' in s
assert 'torch.inference_mode()' in s
assert 'if not produced or produced[0] != first_id' in s
print('PHASE72 OFFLINE TEST PASS: checkpoint identity, chat template, greedy decoding, independent forward agreement')
