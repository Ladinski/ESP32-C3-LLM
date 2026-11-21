"""Smoke test for Phase 36 commands and diagnostic-output contracts (no NumPy)."""
from pathlib import Path

def test():
    root = Path(__file__).resolve().parent
    firmware = (root/'main'/'esp32_c3_llm.c').read_text()
    runner = (root/'run_qwen_stack.py').read_text()
    for command in ('START FP32 DECODE %d','START GROUPED DECODE %d','START FP32 %d','START GROUPED %d'):
        assert command in firmware, command
    assert 'LAYER_FULL %d %d %d' in firmware
    assert 'READY PHASE36' in firmware
    assert 'DONE PHASE36' in firmware
    assert '--kernel' in runner and 'kernel_mode.upper()' in runner
    assert 'LAYER-BY-LAYER FULL ACTIVATION DIAGNOSTICS' in runner
    assert 'DRIFT WARNING' in runner
    print('PHASE 36 PROTOCOL PASS: runtime kernel selection + full-layer diagnostics')

if __name__=='__main__':test()
