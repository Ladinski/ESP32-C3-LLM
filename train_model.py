"""Train a real handwritten-digit MLP, quantize to INT4, export ESP32 weights.

Run: uv run --python 3.12 --with numpy --with scikit-learn python train_model.py
"""
import json
import struct
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "digits_int4_model.bin"
INFO_PATH = ROOT / "digits_int4_metrics.json"
HIDDEN_DIVISOR = 100


def pack_int4(matrix):
    """Pack each output neuron's input weights, low nibble first."""
    arr = np.asarray(matrix, dtype=np.int8)
    assert arr.ndim == 2 and arr.shape[1] % 2 == 0
    assert np.all((-8 <= arr) & (arr <= 7))
    lo = (arr[:, 0::2] & 15).astype(np.uint8)
    hi = ((arr[:, 1::2] & 15).astype(np.uint8) << 4)
    return (lo | hi).tobytes()


def int_forward(inputs, q1, q2, b1, b2):
    x = np.asarray(inputs, dtype=np.int32)
    hidden = x @ q1.astype(np.int32).T + b1
    hidden = np.clip(np.maximum(0, hidden // HIDDEN_DIVISOR), 0, 127)
    return hidden @ q2.astype(np.int32).T + b2


def build_model():
    digits = load_digits()
    raw_train, raw_test, y_train, y_test = train_test_split(
        digits.data.astype(np.uint8), digits.target,
        test_size=0.25, random_state=42, stratify=digits.target,
    )
    classifier = MLPClassifier(
        hidden_layer_sizes=(64,), activation="relu", solver="adam",
        alpha=0.0001, max_iter=400, random_state=42,
    )
    classifier.fit(raw_train.astype(np.float64) / 16.0, y_train)
    w1, w2 = classifier.coefs_  # input x hidden, hidden x output
    b1, b2 = classifier.intercepts_

    s1 = np.max(np.abs(w1)) / 7.0
    s2 = np.max(np.abs(w2)) / 7.0
    q1 = np.clip(np.rint(w1.T / s1), -8, 7).astype(np.int8)
    q2 = np.clip(np.rint(w2.T / s2), -8, 7).astype(np.int8)
    input_scale = 7.0  # each pixel 0..16 maps to integer 0..112
    bias1 = np.rint(b1 * (16.0 * input_scale) / s1).astype(np.int32)
    hidden_unit = HIDDEN_DIVISOR * s1 / (16.0 * input_scale)
    bias2 = np.rint(b2 / (hidden_unit * s2)).astype(np.int32)

    # Binary file: [256 bytes bias1][40 bytes bias2][2048 bytes w1][320 bytes w2].
    # The sample's 64 pixel bytes are prepended at inference time by the host.
    weights = (
        struct.pack("<64i", *bias1.tolist())
        + struct.pack("<10i", *bias2.tolist())
        + pack_int4(q1)
        + pack_int4(q2)
    )
    assert len(weights) == 2664
    MODEL_PATH.write_bytes(weights)

    test_inputs = raw_test.astype(np.int32) * 7
    reference_scores = int_forward(test_inputs, q1, q2, bias1, bias2)
    quantized_accuracy = float(np.mean(reference_scores.argmax(axis=1) == y_test))
    float_accuracy = float(classifier.score(raw_test.astype(np.float64) / 16.0, y_test))
    metrics = {
        "dataset": "sklearn handwritten digits (8x8, ten classes)",
        "architecture": "64 -> ReLU(64) -> 10",
        "weight_format": "signed INT4 packed (biases signed int32)",
        "test_samples": int(len(y_test)),
        "float_test_accuracy": float_accuracy,
        "int4_test_accuracy": quantized_accuracy,
        "split_seed": 42,
        "model_bytes": len(weights),
        "hidden_divisor": HIDDEN_DIVISOR,
    }
    INFO_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print("PHASE 7 | Training completed")
    print("Dataset: handwritten digits, 8x8 pixels, labels 0-9")
    print("Model: 64 -> 64 ReLU -> 10, packed INT4 weights")
    print(f"Float model test accuracy: {float_accuracy:.2%}")
    print(f"Quantized INT4 test accuracy: {quantized_accuracy:.2%}")
    print(f"Held-out test samples: {len(y_test)}")
    print(f"Saved: {MODEL_PATH.name} ({len(weights):,} bytes)")
    print(f"Saved: {INFO_PATH.name}")
    if quantized_accuracy < 0.90:
        raise RuntimeError("Quantized accuracy below 90%; refusing to accept training result")


if __name__ == "__main__":
    build_model()
