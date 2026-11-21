"""Phase 57: offline second-pass Q15 residual correction of one INT4 group.

The ESP32 is NOT involved. This analyzes a previously captured ESP32 activation
vector and the genuine pretrained INT4 matrix. It does not generate model text.
Do not describe this offline result as an on-device accuracy/speed verification.
"""
import argparse
import json
import mmap
from pathlib import Path
import numpy as np

ROWS, COLS, GROUP = 896, 4864, 128
NG = COLS // GROUP
ATOL, RTOL = 0.001, 0.001


def q15(v):
    """Match Phase55's primary Q15 quantization (float32 rounding)."""
    v = np.asarray(v, dtype=np.float32)
    peak = np.max(np.abs(v))
    if peak == 0:
        return np.zeros(v.shape, dtype=np.int16), np.float32(1.0)
    inv = np.float32(np.float32(32760.0) / np.float32(peak))
    step = np.float32(np.float32(peak) / np.float32(32760.0))
    return np.clip(np.rint(v * inv), -32760, 32760).astype(np.int16), step


def make_quantized(x):
    q = np.zeros((NG, GROUP), dtype=np.int16)
    scale = np.empty(NG, dtype=np.float32)
    res = np.zeros((NG, GROUP), dtype=np.float32)
    for g in range(NG):
        v = x[g * GROUP:(g + 1) * GROUP]
        q[g], scale[g] = q15(v)
        res[g] = v - q[g].astype(np.float32) * scale[g]
    return q, scale, res


def select_group(x, residuals):
    # Oracle-free; uses only actual device activations.
    scores = np.sum(np.abs(residuals.astype(np.float64)), axis=1)
    return int(np.argmax(scores)), scores


def unpack_rows(mm, offset, start, stop, row_bytes):
    span = mm[offset + start * row_bytes: offset + stop * row_bytes]
    raw = np.frombuffer(span, dtype=np.uint8)
    expected = (stop - start) * row_bytes
    if raw.size != expected:
        raise ValueError(f'INT4 image truncated at rows {start}:{stop}')
    data = raw.reshape(stop - start, row_bytes)
    lo = (data[:, :COLS // 2] & 15).astype(np.int8)
    hi = (data[:, :COLS // 2] >> 4).astype(np.int8)
    lo[lo >= 8] -= 16
    hi[hi >= 8] -= 16
    w = np.empty((stop - start, COLS), dtype=np.int8)
    w[:, ::2] = lo
    w[:, 1::2] = hi
    scales = np.frombuffer(data[:, COLS // 2:COLS // 2 + 4].copy().tobytes(), dtype='<f4')
    return w, scales


def compute(mm, layout, x, reference):
    offset, rows, cols, row_bytes = layout
    if (rows, cols) != (ROWS, COLS):
        raise ValueError('Unexpected Qwen layer-3 down-projection dimensions')
    primary_q, primary_scale, residual = make_quantized(x)
    chosen, scores = select_group(x, residual)
    rq, rstep = q15(residual[chosen])
    approx = np.zeros(ROWS, dtype=np.float32)
    corrected = np.zeros(ROWS, dtype=np.float32)
    exact = np.zeros(ROWS, dtype=np.float32)
    correction = np.zeros(ROWS, dtype=np.float32)
    row_quant_base = np.zeros(ROWS, dtype=np.float32)
    for start in range(0, ROWS, 16):
        stop = min(start + 16, ROWS)
        w, weight_scales = unpack_rows(mm, offset, start, stop, row_bytes)
        sums = np.zeros(stop - start, dtype=np.float32)
        for g in range(NG):
            chunk = w[:, g*GROUP:(g+1)*GROUP].astype(np.int32)
            dot = chunk @ primary_q[g].astype(np.int32)
            sums += dot.astype(np.float32) * primary_scale[g]
        idx = slice(chosen*GROUP,(chosen+1)*GROUP)
        correction[start:stop] = (w[:, idx].astype(np.int32) @ rq.astype(np.int32)).astype(np.float32) * rstep * weight_scales
        row_quant_base[start:stop] = sums * weight_scales
        corrected[start:stop] = row_quant_base[start:stop] + correction[start:stop]
        # Post-inference FLOAT64 reference, for diagnostics only.
        exact[start:stop] = (w.astype(np.float64) @ reference.astype(np.float64)).astype(np.float32) * weight_scales
    return chosen, scores, primary_scale[chosen], rstep, row_quant_base, corrected, exact, correction


def metrics(predicted, exact):
    delta = np.abs(predicted.astype(np.float64) - exact.astype(np.float64))
    threshold = ATOL + RTOL * np.abs(exact.astype(np.float64))
    return {'max_absolute_error': float(np.max(delta)),
            'mean_absolute_error': float(np.mean(delta)),
            'mixed_tolerance_failures': int(np.count_nonzero(delta > threshold)),
            'max_normalized_error': float(np.max(delta / np.maximum(np.abs(exact),1)))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--trace', default='phase49_act_trace.npz')
    ap.add_argument('--metadata', default='phase25_qwen_24layers_int4.json')
    ap.add_argument('--image', default='phase25_qwen_24layers_int4.bin')
    ap.add_argument('--report', default='phase57_integer_residual_report.json')
    args = ap.parse_args()
    meta = json.loads(Path(args.metadata).read_text(encoding='utf-8'))
    image = Path(args.image)
    if image.stat().st_size != int(meta['image_size']):
        raise ValueError('Pretrained INT4 model image size does not match metadata')
    m = meta['layers'][2]['matrices'][6]
    layout = tuple(int(m[k]) for k in ('offset', 'rows', 'cols', 'row_bytes'))
    with np.load(args.trace, allow_pickle=False) as data:
        device = np.asarray(data['device_ACT'], dtype=np.float32)
        reference = np.asarray(data['reference_ACT'], dtype=np.float32)
    if device.shape != (COLS,) or reference.shape != (COLS,):
        raise ValueError('Expected saved 4864-value device and reference ACT vectors')
    if not np.isfinite(device).all() or not np.isfinite(reference).all():
        raise ValueError('Nonfinite activations')
    with image.open('rb') as fp, mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        group, scores, s0, s1, base, refined, exact, correction = compute(mm, layout, device, reference)
    b, r = metrics(base, exact), metrics(refined, exact)
    improvement = (1 - r['mixed_tolerance_failures'] / max(1, b['mixed_tolerance_failures']))*100
    print('PHASE57 | ONE-GROUP DUAL-Q15 INTEGER RESIDUAL GATE | OFFLINE ONLY')
    print('No COM3, no inference, no firmware changes, original pretrained INT4 weights.')
    print(f'SELECTED GROUP {group}/{NG-1}; residual_L1={scores[group]:.8g}; original_step={s0:.9g}; residual_step={s1:.9g}')
    print(f'BASELINE grouped Q15: max_abs={b["max_absolute_error"]:.9g} mixed_failures={b["mixed_tolerance_failures"]}/{ROWS}')
    print(f'DUAL-Q15 one group: max_abs={r["max_absolute_error"]:.9g} mixed_failures={r["mixed_tolerance_failures"]}/{ROWS}')
    print(f'FAILURE REDUCTION {improvement:.2f}% | estimated extra dot-product work: 1/{NG} groups = {100/NG:.2f}% (NOT a measured runtime)')
    print('Residual correction vector: max_abs=%.9g mean_abs=%.9g' % (np.max(np.abs(correction)), np.mean(np.abs(correction))))
    acceptable = r['mixed_tolerance_failures'] <= 10 and improvement >= 80
    print('PHASE57 OFFLINE GATE ' + ('PASS: candidate for hardware microbenchmark' if acceptable else 'FAIL: discard this candidate; leave firmware unchanged'))
    print('Warning: one saved activation vector cannot guarantee accuracy across other tokens/layers.')
    print('Warning: actual ESP32 int32 overflow checks, q15 scale implementation and row speed remain unverified.')
    report = {'group': group, 'baseline': b, 'candidate': r, 'failure_reduction_percent': improvement,
              'estimated_extra_dot_fraction_not_runtime': 1/NG, 'offline_gate_pass': acceptable,
              'constraints': 'Offline attribution ONLY, no inference, no firmware speed validated'}
    Path(args.report).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('REPORT:', args.report)

if __name__ == '__main__':
    main()
