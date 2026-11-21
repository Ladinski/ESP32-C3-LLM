# Phase 74: original Instruct vs exported INT4 per-layer numerical audit

PC-side diagnostic only. No firmware changes, COM3, ESP32 inference, or writes to the existing model images.

Run from `C:\Projects\ESP32-C3-LLM\esp32_c3_llm\phase73_instruct`, where the Instruct INT4 images already exist.

## Regular PowerShell

```powershell
cd C:\Projects\ESP32-C3-LLM\esp32_c3_llm\phase73_instruct
Expand-Archive "$HOME\Downloads\esp32_phase74_instruct_layer_audit.zip" -DestinationPath . -Force
Copy-Item .\phase74_work\phase74_layer_fidelity.py . -Force
Copy-Item .\phase74_work\test_phase74_offline.py . -Force
..\.phase72-venv\Scripts\python.exe test_phase74_offline.py
..\.phase72-venv\Scripts\python.exe -u phase74_layer_fidelity.py --prompt "Hello" --layers 24
```

Compare all 896 hidden values after each of 24 layers, using the official Qwen2.5-0.5B-Instruct checkpoint in FP32 and the exact row-INT4 NumPy replay. Scores use abs tolerance .001 plus relative .001*abs(reference). Reports first differing layer. Source and model image SHA verified. Uses original model only after standalone model download; no ESP32 model is changed.

**Limitations**: This is a layer-level attribution, not proof that INT4 quantization alone causes mismatch. To separate quantization error from implementation bugs, follow up with a first-layer operator-level original-weight versus INT4 comparison if needed. Baseline model was previously observed to disagree on the first token; this experiment isolates how early divergence starts.

No ESP-IDF terminal is required. For any future firmware builds, first `idf.py fullclean`, then `idf.py build`.
