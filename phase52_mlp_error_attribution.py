"""Phase 52: offline-only counterfactual attribution of layer-3 MLP activation error.

Reads a previously recorded ESP32/NumPy activation trace plus immutable pretrained
INT4 model bytes. Computes W_down @ (ACT_device - ACT_reference) only AFTER a
completed hardware inference; never supplies neural results to the ESP32.
"""
import argparse
import hashlib
import json
import mmap
from pathlib import Path

import numpy as np


def read_mlp_matrix(metadata, index=2):
    layers = metadata['layers']
    if index >= len(layers):
        raise ValueError('Metadata does not contain transformer layer 3')
    m = layers[index]['matrices'][6]  # q,k,v,o,gate,up,down
    rows, cols, row_bytes = (int(m[k]) for k in ('rows', 'cols', 'row_bytes'))
    if rows != 896 or cols != 4864 or row_bytes < (cols // 2 + 4):
        raise ValueError(f'Unexpected Qwen down-projection geometry: {rows=} {cols=} {row_bytes=}')
    return int(m['offset']), rows, cols, row_bytes


def projection_from_int4(mm, matrix, inputs, batch_rows=32):
    """Reference diagnostic, not ESP32 inference. Outputs W@delta for each delta."""
    offset, rows, cols, row_bytes = matrix
    x = np.asarray(inputs, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != cols:
        raise ValueError('Expected N x 4864 counterfactual delta vectors')
    out = np.empty((len(x), rows), dtype=np.float32)
    for start in range(0, rows, batch_rows):
        count = min(batch_rows, rows - start)
        raw = np.frombuffer(mm[offset + start*row_bytes:
                               offset + (start + count)*row_bytes], dtype=np.uint8)
        chunk = raw.reshape(count, row_bytes)
        packed = chunk[:, :cols // 2]
        weights = np.empty((count, cols), dtype=np.float32)
        low = (packed & 0x0f).astype(np.int8)
        high = (packed >> 4).astype(np.int8)
        low[low >= 8] -= 16
        high[high >= 8] -= 16
        weights[:, 0::2] = low
        weights[:, 1::2] = high
        scales = np.frombuffer(chunk[:, cols // 2: cols // 2 + 4].copy().tobytes(), dtype='<f4')
        # Same *quantized* model as the ESP32; PC computes counterfactual only.
        out[:, start:start + count] = (weights @ x.T).T * scales[None, :]
    return out


def attribution(trace, mm, matrix):
    device = np.asarray(trace['device_ACT'], dtype=np.float32)
    reference = np.asarray(trace['reference_ACT'], dtype=np.float32)
    if device.shape != (4864,) or reference.shape != (4864,):
        raise ValueError('Trace must contain 4,864 ACT values')
    if not np.isfinite(device).all() or not np.isfinite(reference).all():
        raise ValueError('Trace contains non-finite activation values')
    delta = device - reference
    order = np.argsort(np.abs(delta))[::-1]
    top3 = order[:3]
    top3_delta = np.zeros_like(delta)
    top3_delta[top3] = delta[top3]
    rest_delta = delta - top3_delta
    # Also test correcting all differences above 0.001, not just three.
    large = np.abs(delta) > np.float32(0.001)
    large_delta = np.where(large, delta, np.float32(0))
    small_delta = delta - large_delta
    labels = ['all_error', 'largest_3', 'remaining_4861', 'error_gt_0.001', 'error_le_0.001']
    projections = projection_from_int4(mm, matrix,
                                     np.stack([delta, top3_delta, rest_delta, large_delta, small_delta]))
    def metrics(v):
        return {
            'max_abs': float(np.max(np.abs(v))),
            'mean_abs': float(np.mean(np.abs(v))),
            'rms': float(np.sqrt(np.mean(np.square(v, dtype=np.float64)))),
            'above_abs_0.001': int(np.count_nonzero(np.abs(v) > 0.001)),
            'above_abs_0.01': int(np.count_nonzero(np.abs(v) > 0.01))
        }
    metrics_by_label = {k: metrics(v) for k, v in zip(labels, projections)}
    # Because down projection is linear: W*delta = W*top3 + W*rest.
    sum_error = float(np.max(np.abs(projections[0] - (projections[1] + projections[2]))))
    norm = float(np.linalg.norm(projections[0].astype(np.float64)))
    largest3_l2 = float(np.linalg.norm(projections[1].astype(np.float64)))
    return {
        'top_three_activation_errors': [
            {'feature': int(i), 'device': float(device[i]), 'reference': float(reference[i]),
             'difference': float(delta[i])} for i in top3
        ],
        'number_activation_errors_over_0.001': int(large.sum()),
        'projection_differences': metrics_by_label,
        'top3_fraction_of_total_projected_l2_norm': (largest3_l2 / norm if norm else None),
        'linearity_check_max_difference': sum_error,
        'observed_prior_phase48_mlp_output_max_error': 0.0100708008,
        'caveat': (
            'These are calculated output differences attributable to the captured ACT mismatch alone. '
            'They do not include the difference introduced by ESP32 grouped-Q15 down-projection arithmetic, '
            'nor reference-state/MLP residual differences. Comparison to the previously logged 0.0100708 '
            'is indicative only; device full MLP_OUT vector is absent from this trace.'
        )
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trace', default='phase49_act_trace.npz')
    p.add_argument('--metadata', default='phase25_qwen_24layers_int4.json')
    p.add_argument('--image', default='phase25_qwen_24layers_int4.bin')
    p.add_argument('--report', default='phase52_mlp_error_attribution.json')
    p.add_argument('--verify-sha', action='store_true', help='Optional full 180MB image SHA256 verification')
    a = p.parse_args()
    meta = json.loads(Path(a.metadata).read_text(encoding='utf-8'))
    matrix = read_mlp_matrix(meta)
    path = Path(a.image)
    if path.stat().st_size != int(meta['image_size']):
        raise ValueError('Model image size does not match metadata')
    if a.verify_sha:
        with path.open('rb') as file:
            sha = hashlib.file_digest(file, 'sha256').hexdigest()
        if sha != meta['sha256']:
            raise ValueError('Model checksum mismatch')
        print('CHECKPOINT SHA256 VERIFIED', sha)
    print('PHASE52 | LAYER-3 MLP DOWN-PROJECTION ERROR ATTRIBUTION')
    print('OFFLINE ONLY. No COM3, no ESP32 flashing, no change to inference performance.')
    print('Actual INT4 layer-3 down weights, prior ESP32 ACT trace; no host-assisted prediction.')
    with np.load(a.trace, allow_pickle=False) as trace, path.open('rb') as file:
        with mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            report = attribution(trace, mm, matrix)
    for feature in report['top_three_activation_errors']:
        print('  ACT feature {feature}: device={device:.9g} ref={reference:.9g} delta={difference:.9g}'.format(**feature))
    print('ACT values with absolute error >0.001:', report['number_activation_errors_over_0.001'])
    print('Projected hidden-state difference caused by activation error ONLY (896 outputs):')
    for name, v in report['projection_differences'].items():
        print(f'  {name:17s} max={v["max_abs"]:.9g} mean={v["mean_abs"]:.9g} '
              f'rms={v["rms"]:.9g} >0.001={v["above_abs_0.001"]} >0.01={v["above_abs_0.01"]}')
    fraction = report['top3_fraction_of_total_projected_l2_norm']
    print('Largest-three projected L2 / all-error projected L2:', 'undefined' if fraction is None else f'{fraction:.4f}')
    print('Linearity consistency:', f'{report["linearity_check_max_difference"]:.6g}')
    if report['linearity_check_max_difference'] > 1e-4:
        print('CAUTION: unusually large counterfactual rounding mismatch')
    print('PREVIOUS ESP32 layer-3 MLP_OUT max error: 0.0100708008 (from Phase 48)')
    print('IMPORTANT: Counterfactual is not a substitute for a paired full MLP_OUT capture.')
    Path(a.report).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('REPORT', a.report)


if __name__ == '__main__':
    main()
