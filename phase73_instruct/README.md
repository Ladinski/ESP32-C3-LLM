# Phase 80: INT8 export and integrity verification (PC only)

This package contains `phase80_export_int8.py`, a separate INT8 model-image exporter, and `test_phase80_offline.py`.

It requires `export_qwen_layer0.py` from Phase 73 in the same directory, plus the `.phase72-venv` environment from Phase 72.

The exporter creates two **new** output images in `phase80_int8_images/`:
- `qwen25_instruct_transformer_w8.bin`: 24-layer signed, symmetric per-row INT8 projection weights + float32 scales, norms and attention biases.
- `qwen25_instruct_head_bf16.bin`: float32 final RMSNorm followed by the original BF16 tied output-head rows.
- `qwen25_instruct_w8_manifest.json`: offsets, dimensions, file hashes and chat prompt token IDs.

These are binary transfer images for a FUTURE firmware implementation. Existing INT4 firmware does NOT understand QW8S, and these images must NOT be flashed or substituted for INT4 files.

## Regular PowerShell

```powershell
cd C:\Projects\ESP32-C3-LLM\esp32_c3_llm\phase73_instruct
Expand-Archive "$HOME\Downloads\esp32_phase80_int8_export.zip" -DestinationPath . -Force
..\.phase72-venv\Scripts\python.exe test_phase80_offline.py
..\.phase72-venv\Scripts\python.exe -u phase80_export_int8.py --layers 24 --prompt "Hello"
..\.phase72-venv\Scripts\python.exe -u phase80_export_int8.py --layers 24 --verify-only
```

The exporter will refuse to overwrite existing Phase 80 images. Re-run with `--verify-only`. If necessary, use a fresh `--output-dir`.

NOTE: The head image deliberately uses BF16 rather than INT8 so this step does not introduce an additional untested head-quantization loss. Matching Phase 79 output still requires ESP32-side integer kernels and correct chat/KV handling.

No serial monitor, COM3, ESP-IDF build or flash is required in Phase 80.
