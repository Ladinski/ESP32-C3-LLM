"""Phase 42: offline lossless compression feasibility of actual pretrained INT4 matrix bytes.

Pure read-only analysis; no device traffic, firmware changes, inference, or model conversion.
A pass only authorizes further investigation, NOT a claim of faster ESP32 inference.
"""
import argparse
import hashlib
import json
import mmap
import statistics
import zlib
from pathlib import Path

MIB = 1024 * 1024


def inspect(args):
    model = Path(args.image)
    meta_path = Path(args.metadata)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("model") != "Qwen/Qwen2.5-0.5B":
        raise ValueError("Not Qwen2.5-0.5B metadata")
    if model.stat().st_size != meta.get("image_size"):
        raise ValueError("Model size does not match metadata")
    if len(meta.get("layers", [])) != 24:
        raise ValueError("Expected complete 24-layer metadata")
    if args.verify_sha:
        with model.open("rb") as f:
            got = hashlib.file_digest(f, "sha256").hexdigest()
        if got != meta.get("sha256"):
            raise ValueError("Model SHA256 does not match metadata")
        print("MODEL SHA256 VERIFIED:", got)
    # Select matrices from the beginning, middle and end to avoid a biased early-layer sample.
    chosen = sorted(set([0, 1, 5, 11, 17, 23]))
    samples = []
    details = []
    with model.open("rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        for layer_index in chosen:
            layer = meta["layers"][layer_index]
            matrices = layer.get("matrices", [])
            if len(matrices) != 7:
                raise ValueError(f"Layer {layer_index} does not have 7 matrices")
            for matrix_index in (0, 3, 4, 6):
                matrix = matrices[matrix_index]
                offset = int(matrix["offset"])
                n = int(matrix["rows"])*int(matrix["row_bytes"])
                if offset < 0 or n <= 0 or offset+n > len(mm):
                    raise ValueError("Bad matrix bounds")
                # Diverse positions within each pretrained tensor, all capped to the same bounded read.
                sample_len = min(args.sample_kib*1024, n)
                sample_offset = offset + (n - sample_len)//2
                raw = bytes(mm[sample_offset:sample_offset+sample_len])
                samples.append(raw)
                details.append((layer_index, matrix_index, len(raw)))
    print(f"REAL INT4 MODEL: {model.name} | sampled {len(samples)} matrix regions | {sum(map(len,samples))/MIB:.2f} MiB")
    print("No model inference, no PC neural computation, no firmware changes, USB Type-C retained.")
    results = []
    for block_size in (4096, 16384, 65536):
        total_raw = total_compressed = 0
        ratios = []
        for raw in samples:
            for begin in range(0, len(raw), block_size):
                block = raw[begin:begin+block_size]
                packed = zlib.compress(block, level=1)
                total_raw += len(block)
                total_compressed += len(packed)
                ratios.append(len(packed)/len(block))
                # Lossless roundtrip validation, so only exact original bytes are acceptable.
                if zlib.decompress(packed) != block:
                    raise AssertionError("Lossless roundtrip failure")
        ratio = total_compressed/total_raw
        savings = (1-ratio)*100
        results.append(dict(block_bytes=block_size, raw_bytes=total_raw,
                            compressed_bytes=total_compressed,
                            ratio=round(ratio, 6), savings_percent=round(savings, 3)))
        print(f"  zlib level1, independent {block_size//1024}KiB blocks: {ratio:.4f}x wire size | {savings:+.2f}% bytes saved | median block ratio {statistics.median(ratios):.4f}x")
    best = max(results, key=lambda x: x["savings_percent"])
    print(f"BEST OBSERVED SAMPLE SAVINGS: {best['savings_percent']:.2f}% using {best['block_bytes']//1024}KiB blocks")
    if best["savings_percent"] < args.minimum_savings:
        print(f"PHASE42 GATE FAIL: <{args.minimum_savings:.0f}% sample savings. REJECT compression; no firmware/inference test.")
    else:
        print(f"PHASE42 ANALYSIS GATE PASS: >={args.minimum_savings:.0f}% sample savings. ESP32 decompression speed/memory must still be benchmarked before adoption.")
    report = dict(model_sha256=meta.get("sha256"), sample_regions=details,
                  sample_bytes=sum(map(len,samples)), results=results,
                  gate_percent=args.minimum_savings, analysis_only=True)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("REPORT:", args.json)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", default="phase25_qwen_24layers_int4.bin")
    p.add_argument("--metadata", default="phase25_qwen_24layers_int4.json")
    p.add_argument("--sample-kib", type=int, default=256)
    p.add_argument("--minimum-savings", type=float, default=25.0)
    p.add_argument("--verify-sha", action="store_true")
    p.add_argument("--json", default="phase42_compression_report.json")
    args = p.parse_args()
    if not 16 <= args.sample_kib <= 4096:
        p.error("--sample-kib must be in [16,4096]")
    if not 0 < args.minimum_savings <= 100:
        p.error("--minimum-savings must be in (0,100]")
    inspect(args)


if __name__ == "__main__":
    main()
