#!/usr/bin/env python3
"""Phase 13: download and convert pretrained GPT-Neo TinyStories-1M.

The output is an interchange format for a future ESP32 runtime, NOT yet
an executable ESP32 model. Computation in this phase occurs on the PC.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np
import torch
from huggingface_hub import hf_hub_download

REPO = 'roneneldan/TinyStories-1M'
EXPECTED_ARCH = 'GPTNeoForCausalLM'


def safe_name(s):
    return re.sub(r'[^A-Za-z0-9_.-]', '_', s)


def quantize_rows(w):
    """Symmetric signed INT4, one float32 scale per row; returns packed bytes."""
    m = w.detach().to(torch.float32).cpu().contiguous().numpy()
    assert m.ndim == 2
    nrows, ncols = m.shape
    if not np.isfinite(m).all():
        raise ValueError('nonfinite matrix values')
    absmax = np.max(np.abs(m), axis=1)
    scale = np.where(absmax == 0.0, 1.0, absmax / 7.0).astype('<f4')
    q = np.rint(m / scale[:, None]).clip(-7, 7).astype(np.int8)
    # Each row is padded separately when column count is odd.
    if ncols % 2:
        q = np.pad(q, ((0, 0), (0, 1)))
    lo = q[:, 0::2].astype(np.int16) & 15
    hi = q[:, 1::2].astype(np.int16) & 15
    packed = (lo | (hi << 4)).astype(np.uint8).tobytes()
    decoded = q[:, :ncols].astype(np.float32) * scale[:, None]
    diff = decoded - m
    return packed, scale.tobytes(), {
        'mae': float(np.mean(np.abs(diff))),
        'rmse': float(np.sqrt(np.mean(diff * diff))),
        'max_abs_error': float(np.max(np.abs(diff))),
        'relative_rmse': float(np.sqrt(np.sum(diff * diff) / max(float(np.sum(m * m)), 1e-12)))
    }


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', default='phase13_export', help='Export directory')
    ap.add_argument('--cache-dir', default=None)
    args = ap.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    print('PHASE 13 | PRETRAINED GPT-NEO IMPORT')
    print('Source:', REPO)
    print('Downloading authentic checkpoint on PC (first run needs internet).')
    cfg_path = hf_hub_download(REPO, filename='config.json', cache_dir=args.cache_dir)
    checkpoint_path = hf_hub_download(REPO, filename='pytorch_model.bin', cache_dir=args.cache_dir)
    config = json.loads(Path(cfg_path).read_text(encoding='utf-8'))
    if EXPECTED_ARCH not in config.get('architectures', []) or config.get('model_type') != 'gpt_neo':
        raise RuntimeError('Unexpected checkpoint architecture; refusing to export')
    # Never execute arbitrary pickle objects; requires a recent torch release.
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if not isinstance(state, dict) or not state:
        raise RuntimeError('Checkpoint does not contain tensor state dict')
    tensors = {k: v for k, v in state.items() if isinstance(v, torch.Tensor)}
    if len(tensors) != len(state):
        raise RuntimeError('Unexpected non-tensor checkpoint entry')
    keys = list(tensors)
    if not any('wte.weight' in k for k in keys):
        raise RuntimeError('Missing token embedding tensor')
    if not any('attn.attention.q_proj.weight' in k for k in keys):
        raise RuntimeError('Missing GPT-Neo attention weight tensor')

    metadata = {
        'format': 'esp32c3-phase13-v1',
        'status': 'PC_EXPORTED_NOT_ESP32_EXECUTABLE',
        'source_repo': REPO,
        'source_config': config,
        'checkpoint_sha256': sha256(checkpoint_path),
        'matrix_encoding': 'signed_int4_symmetric_per_row_scale_fp32',
        'matrix_packing': 'low_nibble_first_two_complement_row_padded',
        'other_tensors': 'float16',
        'tensors': [],
        'skipped_nonparameter_buffers': [],
        'parameter_count_unique_checkpoint_tensors': 0,
        'export_bytes': 0,
    }
    matrix_stats = []
    for i, (name, value) in enumerate(sorted(tensors.items())):
        # GPT-Neo's fixed negative masking sentinel is not a learned parameter.
        # Typical value is -1e9, which overflows float16. The eventual runtime
        # must apply its own causal mask, rather than load this checkpoint buffer.
        if name.endswith('.attn.attention.masked_bias'):
            if not torch.is_floating_point(value) or value.numel() != 1:
                raise RuntimeError(f'Unexpected masked_bias tensor: {name} {value.shape} {value.dtype}')
            sentinel = float(value.detach().reshape(-1)[0].item())
            if not math.isfinite(sentinel) or sentinel >= -65504.0:
                raise RuntimeError(f'Unexpected masked_bias value: {name} {sentinel}')
            metadata['skipped_nonparameter_buffers'].append({
                'name': name, 'dtype': str(value.dtype), 'shape': list(value.shape),
                'reason': 'fixed_attention_masking_sentinel_recreated_by_runtime'
            })
            print('Skipping fixed masking sentinel:', name)
            continue
        # GPT-Neo checkpoints include a precomputed boolean lower-triangular
        # causal-attention mask. This is a model buffer, not a learned weight.
        # Recreate it in the eventual attention runtime; do not quantize it.
        if not torch.is_floating_point(value):
            if name.endswith('.attn.attention.bias') and value.dtype == torch.bool:
                metadata['skipped_nonparameter_buffers'].append({
                    'name': name, 'dtype': str(value.dtype),
                    'shape': list(value.shape), 'reason': 'causal_attention_mask_recreated_by_runtime'
                })
                print('Skipping fixed causal-attention mask:', name)
                continue
            raise RuntimeError(f'Unexpected nonfloating tensor: {name} {value.dtype}')
        fp32 = value.detach().to(torch.float32).cpu()
        if not torch.isfinite(fp32).all():
            raise RuntimeError(f'Nonfinite values: {name}')
        base = f'{i:03d}_{safe_name(name)}'
        item = {'name': name, 'shape': list(fp32.shape), 'original_dtype': str(value.dtype), 'numel': int(fp32.numel())}
        metadata['parameter_count_unique_checkpoint_tensors'] += fp32.numel()
        if fp32.ndim == 2:
            packed, scales, err = quantize_rows(fp32)
            data_file = base + '.int4'
            scale_file = base + '.scale_f32'
            (out / data_file).write_bytes(packed)
            (out / scale_file).write_bytes(scales)
            item.update(encoding='int4_per_row', file=data_file, scale_file=scale_file,
                        scale_dtype='float32_le', bytes=len(packed) + len(scales), error=err)
            matrix_stats.append((name, err['relative_rmse']))
        else:
            if float(fp32.abs().max().item()) > 65504.0:
                raise RuntimeError(f'FP16 range exceeded by tensor: {name}')
            arr = fp32.numpy().astype('<f2')
            data_file = base + '.fp16'
            (out / data_file).write_bytes(arr.tobytes(order='C'))
            item.update(encoding='float16_le', file=data_file, bytes=arr.nbytes)
        item['sha256'] = sha256(out / data_file)
        metadata['tensors'].append(item)
        metadata['export_bytes'] += item['bytes']

    (out / 'config.json').write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
    (out / 'manifest.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    summary = {
        'source': REPO,
        'architecture': config['architectures'],
        'layers': config.get('num_layers'),
        'hidden_size': config.get('hidden_size'),
        'heads': config.get('num_heads'),
        'vocabulary': config.get('vocab_size'),
        'context_limit': config.get('max_position_embeddings'),
        'attention_layers': config.get('attention_layers'),
        'parameter_count': metadata['parameter_count_unique_checkpoint_tensors'],
        'export_bytes': metadata['export_bytes'],
        'tensor_count': len(metadata['tensors']),
        'skipped_nonparameter_buffer_count': len(metadata['skipped_nonparameter_buffers']),
        'worst_relative_matrix_rmse': max((x[1] for x in matrix_stats), default=None),
        'checkpoint_sha256': metadata['checkpoint_sha256'],
        'status': metadata['status'],
    }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print('\nPASS: checkpoint downloaded, parsed and tensors exported.')
    print('IMPORTANT: this is not runnable ESP32 firmware yet.')
    print('Export:', out)


if __name__ == '__main__':
    main()
