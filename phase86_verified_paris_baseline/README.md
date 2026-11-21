# Phase 86: Verified PC-assisted ESP32-C3 LLM demonstration

Prompt: `Capital of France:`
Tokenizer IDs: `[63593, 315, 9625, 25]`
ESP32-generated next token: `12095` (` Paris`)
Decoded output: `Capital of France: Paris`
Device result: `PHASE85 DEVICE GATE PASS prompt_tokens=4 generated_tokens=1 layers=24`

All 24 transformer layers, KV cache/attention and output-head token selection ran on the device. PC served embeddings/weights and performed offline validation. Full layer-by-layer independent reference validation was NOT performed; first-layer checks at four positions and the output-head audit passed.

This archive contains source, a JSON run report, checksums, configuration (if found), and the reference model export manifest. It intentionally DOES NOT contain massive weight images, virtual environments, builds, or credentials. Original exported model files must remain in the project's `phase73_instruct/phase80_int8_images` directory for reproduction.

No firmware rebuild, flash, USB connection, or additional inference run is needed to create this archive.
