# ESP32-C3 LLM

An ESP-IDF research project that runs quantized transformer inference on an ESP32-C3 with a PC streaming model data over USB Serial/JTAG. The latest verified demo processed the fixed prompt `Capital of France:` through all 24 layers of Qwen2.5-0.5B-Instruct and selected the next token ` Paris` (ID 12095) on the device.

## Verified result

| Item | Recorded result |
| --- | --- |
| Prompt token IDs | 63593, 315, 9625, 25 |
| Decoded output | `Capital of France: Paris` |
| Device work | 24 transformer layers, causal grouped-query attention and KV cache, and output-head token selection |
| PC work | Tokenization, streaming embeddings and weights, and reference checks after inference |
| Reference coverage | First layer at each of the four prompt positions and the output head |
| Recorded elapsed time | About 69 minutes for the run |

The Phase 85 runner is a fixed-prompt, single-next-token demonstration. The reported checks cover the first layer and output head. A full independent comparison of all 24 layers and general chat generation are outside the scope of this recorded run. See the [run report](phase85_paris_prompt_report.json) and [baseline notes](phase86_verified_paris_baseline/README.md).

## Project layout

| Path | Purpose |
| --- | --- |
| [main/](main/) | Active ESP32-C3 firmware and ESP-IDF component definition |
| [sdkconfig](sdkconfig) | ESP32-C3 project configuration with 4 MB flash and USB Serial/JTAG console |
| [phase85_paris_prompt.py](phase85_paris_prompt.py) | Host driver and first-layer/output-head validation for the Paris demo |
| [phase73_instruct/](phase73_instruct/) | Instruct-model fidelity experiments, INT8 export, and offline checks |
| [phase86_verified_paris_baseline/](phase86_verified_paris_baseline/) | Frozen source, configuration, export manifest, run report, and checksums |
| [phase86_tools/](phase86_tools/) | PowerShell tooling to preserve the verified baseline |
| `run_*.py`, `bench_*.py`, `test_*.py` and earlier phase scripts | Historical host runners, benchmarks, and regression checks. Match their firmware and data formats before running them. |

## Reproduce the fixed-prompt demo

Use an ESP32-C3 board with native USB Serial/JTAG exposed, a USB data cable, an ESP-IDF PowerShell environment with `idf.py`, and Python 3.12 on the PC. The examples use Windows PowerShell and `COM3`. Substitute the port assigned to your board. Allow several gigabytes of disk space for the original checkpoint, Hugging Face cache, and exported images.

### 1. Set up the host and export weights

Run from the repository root in regular PowerShell.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install numpy pyserial "transformers>=4.45,<5" "huggingface_hub<1" safetensors
.\.venv\Scripts\python.exe phase73_instruct/test_phase80_offline.py
.\.venv\Scripts\python.exe -u phase73_instruct/phase80_export_int8.py --layers 24 --output-dir phase73_instruct/phase80_int8_images
.\.venv\Scripts\python.exe -u phase73_instruct/phase80_export_int8.py --layers 24 --output-dir phase73_instruct/phase80_int8_images --verify-only
```

The first export uses the official Qwen2.5-0.5B-Instruct checkpoint and tokenizer. It writes a QW8S-v1 INT8 transformer image with per-row float32 scales, a BF16 output-head image, and a JSON manifest with offsets and SHA-256 hashes. The model images are streamed by the PC. The exporter refuses to overwrite existing images. Use `--verify-only` for a complete existing export or a fresh `--output-dir` with the runner’s `--manifest` option.

### 2. Build and flash

From an ESP-IDF PowerShell at the repository root, build with the checked-in ESP32-C3 configuration and flash the board.

```powershell
idf.py fullclean
idf.py build
idf.py -p COM3 flash
```

### 3. Run the host driver

Close any serial monitor before starting the host driver. The driver reads the checkpoint and tokenizer from the local Hugging Face cache. Use the same Windows account and cache location as for the export.

```powershell
.\.venv\Scripts\python.exe -u phase85_paris_prompt.py --prepare-only
.\.venv\Scripts\python.exe -u phase85_paris_prompt.py --port COM3 --timeout 14400 --report paris_run.json
```

`--prepare-only` checks the host-side inputs without opening the device. The device run should report `PHASE85 DEVICE GATE PASS prompt_tokens=4 generated_tokens=1 layers=24` and write a report with the selected token and reference-check results. For serial connection failures, confirm the COM port and close any process that owns it. For an image integrity failure, check that the manifest and both binaries came from the same export.

The [Phase 70 host-protocol notes](README_FIX.md) describe an earlier INT4 baseline. The Phase 72 model-quality diagnostic in `phase72_model_quality_diagnostic.py` compares the base and Instruct checkpoints on a PC. Those phases use separate workflows from the Phase 85 demo.
