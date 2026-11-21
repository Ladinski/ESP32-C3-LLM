"""Send actual held-out digit images and trained INT4 weights to the ESP32-C3.

Run: uv run --python 3.12 --with numpy --with scikit-learn --with pyserial python digit_inference.py
"""
import argparse
import json
import struct
import time
from pathlib import Path

import numpy as np
import serial
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "digits_int4_model.bin"
INFO_PATH = ROOT / "digits_int4_metrics.json"
FNV_OFFSET = 2166136261
FNV_PRIME = 16777619


def fnv32(data):
    result = FNV_OFFSET
    for b in data:
        result = ((result ^ b) * FNV_PRIME) & 0xFFFFFFFF
    return result


def unpack_int4(data, rows, cols):
    raw = np.frombuffer(data, dtype=np.uint8).reshape(rows, cols // 2)
    low = (raw & 15).astype(np.int32)
    high = (raw >> 4).astype(np.int32)
    low[low >= 8] -= 16
    high[high >= 8] -= 16
    result = np.empty((rows, cols), dtype=np.int32)
    result[:, ::2], result[:, 1::2] = low, high
    return result


def decode_model(blob):
    if len(blob) != 2664:
        raise RuntimeError(f"Expected 2664 model bytes, found {len(blob)}")
    bias1 = np.asarray(struct.unpack("<64i", blob[:256]), dtype=np.int32)
    bias2 = np.asarray(struct.unpack("<10i", blob[256:296]), dtype=np.int32)
    w1 = unpack_int4(blob[296:2344], 64, 64)
    w2 = unpack_int4(blob[2344:], 10, 64)
    return bias1, bias2, w1, w2


def python_forward(x, model):
    b1, b2, w1, w2 = model
    h = np.clip(np.maximum(0, (w1 @ x.astype(np.int32) + b1) // 100), 0, 127)
    return w2 @ h + b2


def expect(port, prefix, timeout=10):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        raw = port.readline()
        if not raw:
            continue
        line = raw.decode("ascii", errors="replace").strip()
        if line.startswith("ERR "):
            raise RuntimeError(f"ESP32: {line}")
        if line.startswith(prefix):
            return line
    raise TimeoutError(f"Expected {prefix!r} from ESP32")


def transact(port, sample, blob, model):
    input_bytes = (sample.astype(np.uint8) * 7).tobytes()
    # Input (64), first biases (256), second biases (40), first weights
    # (4 chunks x 512), second weights (320).
    chunks = [input_bytes, blob[:256], blob[256:296]]
    chunks += [blob[296 + i * 512:296 + (i + 1) * 512] for i in range(4)]
    chunks.append(blob[2344:])
    payload = b"".join(chunks)
    assert len(payload) == 2728

    port.write(b"RUN\n")
    go = expect(port, "GO ")
    if go != f"GO {len(payload)}":
        raise RuntimeError(f"Unexpected GO response: {go}")

    transferred = 0
    started = time.perf_counter()
    for chunk in chunks:
        port.write(chunk)
        transferred += len(chunk)
        if transferred != len(payload):
            ack = expect(port, "ACK ")
            if ack != f"ACK {transferred}":
                raise RuntimeError(f"Incorrect ACK: {ack}, expected ACK {transferred}")
    out = expect(port, "OUT ")
    done = expect(port, "DONE ")
    elapsed = time.perf_counter() - started
    result = np.asarray([int(x) for x in out.split()[1:]], dtype=np.int32)
    if len(result) != 10:
        raise RuntimeError(f"Expected ten scores, got {len(result)}")

    parts = done.split()
    if len(parts) != 8:
        raise RuntimeError(f"Unexpected DONE: {done}")
    _, count, received_hash, chosen, compute, validation, receive, device = parts
    if int(count) != len(payload) or int(received_hash, 16) != fnv32(payload):
        raise RuntimeError("Payload length/hash mismatch")
    if int(chosen) != int(result.argmax()):
        raise RuntimeError("Device winner does not match device scores")

    reference = python_forward(sample.astype(np.int32) * 7, model)
    if not np.array_equal(result, reference):
        raise RuntimeError(f"CPU mismatch! ESP32={result.tolist()} Python={reference.tolist()}")
    return result, elapsed, tuple(int(v) for v in (compute, validation, receive, device))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", default="COM3")
    p.add_argument("--samples", type=int, default=100)
    args = p.parse_args()
    if args.samples < 1 or args.samples > 450:
        p.error("--samples must be between 1 and 450")

    if not MODEL_PATH.exists():
        raise SystemExit("Model missing. Run train_model.py first.")
    blob = MODEL_PATH.read_bytes()
    model = decode_model(blob)
    metrics = json.loads(INFO_PATH.read_text(encoding="utf-8"))
    digits = load_digits()
    _, x_test, _, y_test = train_test_split(
        digits.data.astype(np.uint8), digits.target,
        test_size=0.25, random_state=42, stratify=digits.target,
    )
    count = min(args.samples, len(y_test))
    print("ESP32-C3 Phase 7 | REAL trained digit classifier")
    print("Model: INT4 64 -> 64 ReLU -> 10; ESP32 executes all math")
    print(f"Training reference INT4 accuracy: {metrics['int4_test_accuracy']:.2%} on {len(y_test)} held-out samples")
    print(f"Testing ESP32 on {count} held-out digit images via {args.port}")

    correct = 0
    times = []
    with serial.Serial(args.port, 115200, timeout=0.25, write_timeout=10) as esp:
        time.sleep(0.3)
        esp.reset_input_buffer()
        esp.write(b"HELLO\n")
        print(expect(esp, "READY PHASE7"))
        for index in range(count):
            scores, elapsed, timings = transact(esp, x_test[index], blob, model)
            predicted = int(scores.argmax())
            actual = int(y_test[index])
            correct += predicted == actual
            times.append(elapsed)
            if index < 10 or (index + 1) % 25 == 0 or index + 1 == count:
                print(f"[{index + 1:3}/{count}] true={actual} predicted={predicted} "
                      f"match=PASS elapsed={elapsed * 1000:.1f}ms")
    accuracy = correct / count
    print("\nALL DEVICE SCORES EXACTLY MATCH PYTHON INT32 REFERENCE")
    print(f"Classification accuracy on {count} images: {accuracy:.2%} ({correct}/{count})")
    print(f"Average transfer + inference: {np.mean(times) * 1000:.2f} ms/image")
    print("PHASE 7 TRAINED INT4 INFERENCE COMPLETE")


if __name__ == "__main__":
    main()
