# Phase 70: Python-only serial ordering fix

No firmware change, no ESP-IDF build, no flash. Existing Phase68 firmware emits `TOKEN 0 COMPLETE` **before** 56 `OUT 0 <start>` records, and repeats that order for token 1. The original Phase70 runner incorrectly demanded those `OUT` records before `TOKEN COMPLETE` and failed after token 0.

Extract this ZIP in project root using **Regular PowerShell**:

```powershell
cd C:\Projects\ESP32-C3-LLM\esp32_c3_llm
Expand-Archive "$HOME\Downloads\esp32_phase70_serial_order_FIX.zip" -DestinationPath . -Force
uv run --python 3.12 python test_phase70_order_fix.py
uv run --python 3.12 python test_phase70_offline.py
```

Close the ESP-IDF monitor, then in **Regular PowerShell**:

```powershell
uv run --python 3.12 --with numpy --with pyserial --with tokenizers --with huggingface_hub python -u phase70_two_token_verify.py --prompt "Hi?" --port COM3 --timeout 10800
```

The fix validates 24 `LAYER` records before each `TOKEN COMPLETE`, requires the complete 896-channel OUT stream before accepting the next token embedding and after inference DONE, and validates both token positions. If it fails on device or reference, retain the logs and do not claim full-model success.
