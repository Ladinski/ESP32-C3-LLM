# Phase 86: Freeze the verified Paris demo

This is a preservation step, not another inference run. Run from **Regular PowerShell**, serial monitor **CLOSED**. No ESP-IDF shell, build, flash or ESP32 required.

Extract this ZIP to `C:\Projects\ESP32-C3-LLM\esp32_c3_llm\phase86_tools`, then run:

```powershell
cd C:\Projects\ESP32-C3-LLM\esp32_c3_llm
.\phase86_tools\phase86_freeze_baseline.ps1
```

This checks the locally generated `phase85_paris_prompt_report.json` and archives firmware, host source, export manifest, any available build configuration, and SHA-256 checksums, intentionally excluding model weight binaries.
