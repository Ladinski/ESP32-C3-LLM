#!/usr/bin/env python3
"""Verify every exported file and numerically check INT4 samples."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dir', default='phase13_export')
    args = p.parse_args()
    root = Path(args.dir)
    manifest = json.loads((root / 'manifest.json').read_text())
    total = 0
    for item in manifest['tensors']:
        path = root / item['file']
        raw = path.read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        if actual != item['sha256']:
            raise RuntimeError('HASH MISMATCH: ' + item['file'])
        n = int(np.prod(item['shape']))
        if item['encoding'] == 'int4_per_row':
            rows, cols = item['shape']
            expected = rows * ((cols + 1) // 2)
            if len(raw) != expected:
                raise RuntimeError('Packed size mismatch: ' + item['name'])
            scales = np.fromfile(root / item['scale_file'], dtype='<f4')
            if len(scales) != rows or not np.isfinite(scales).all() or (scales <= 0).any():
                raise RuntimeError('Invalid scale: ' + item['name'])
            sample_rows = [0, rows // 2, rows - 1]
            packed = np.frombuffer(raw, dtype=np.uint8).reshape(rows, -1)[sample_rows]
            qlo = (packed & 15).astype(np.int16)
            qhi = (packed >> 4).astype(np.int16)
            qlo[qlo >= 8] -= 16
            qhi[qhi >= 8] -= 16
            decoded = np.empty((len(sample_rows), packed.shape[1] * 2), dtype=np.int16)
            decoded[:, 0::2] = qlo
            decoded[:, 1::2] = qhi
            decoded = decoded[:, :cols].astype(np.float32) * scales[sample_rows, None]
            if not np.isfinite(decoded).all():
                raise RuntimeError('Nonfinite decoded weight: ' + item['name'])
        elif item['encoding'] == 'float16_le':
            if len(raw) != n * 2:
                raise RuntimeError('FP16 length mismatch: ' + item['name'])
            if not np.isfinite(np.frombuffer(raw, dtype='<f2')).all():
                raise RuntimeError('Nonfinite FP16: ' + item['name'])
        else:
            raise RuntimeError('Unknown tensor encoding: ' + item['name'])
        total += item['bytes']
    if total != manifest['export_bytes']:
        raise RuntimeError('Export total mismatch')
    print(f"PASS: {len(manifest['tensors'])} tensors, {total:,} bytes, hashes and sample INT4 decode verified")
    print('Checkpoint source:', manifest['source_repo'])
    print('Status:', manifest['status'])


if __name__ == '__main__':
    main()
