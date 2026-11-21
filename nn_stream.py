"""Phase 6: send packed INT4 weights; all neural-network math runs on ESP32."""
from pathlib import Path
import time
import serial

PORT = "COM3"
BLOCK_BYTES = 512
TOTAL_BYTES = 8704
FILE = Path("two_layer_int4_weights.bin")
FNV_OFFSET = 2166136261
FNV_PRIME = 16777619


def make_weights():
    # Deterministic synthetic weights; the PC does not evaluate the network.
    state = 0x12345678
    data = bytearray(TOTAL_BYTES)
    for i in range(TOTAL_BYTES):
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= state >> 17
        state ^= (state << 5) & 0xFFFFFFFF
        state &= 0xFFFFFFFF
        data[i] = state & 0xFF
    FILE.write_bytes(data)
    return bytes(data)


def fnv32(data):
    value = FNV_OFFSET
    for byte in data:
        value = ((value ^ byte) * FNV_PRIME) & 0xFFFFFFFF
    return value


def expect(esp, prefix):
    limit = time.monotonic() + 10
    while time.monotonic() < limit:
        line = esp.readline().decode("ascii", errors="replace").strip()
        if not line:
            continue
        if line.startswith("ERR"):
            raise RuntimeError(f"ESP32: {line}")
        if line.startswith(prefix):
            return line
    raise TimeoutError(f"Waiting for {prefix!r}")


def main():
    weights = make_weights()
    expected_hash = f"{fnv32(weights):08x}"
    print("ESP32-C3 Phase 6 | Two-layer INT4 neural network")
    print("256 inputs -> 64 ReLU activations -> 16 output scores")
    print(f"Weight data: {len(weights):,} bytes; host FNV32: {expected_hash}")
    print("PC: weight generation / transport only; ESP32: all neural-network math")

    with serial.Serial(PORT, 115200, timeout=0.25, write_timeout=10) as esp:
        time.sleep(1)
        esp.reset_input_buffer()
        esp.write(b"HELLO\n")
        print(expect(esp, "READY"))
        esp.write(b"RUN\n")
        go = expect(esp, "GO ")
        if go != f"GO {TOTAL_BYTES}":
            raise RuntimeError(f"Unexpected firmware dimensions: {go}")

        start = time.perf_counter()
        sent = 0
        while sent < TOTAL_BYTES:
            block = weights[sent:sent + BLOCK_BYTES]
            if len(block) != BLOCK_BYTES:
                raise RuntimeError("Unexpected partial block")
            esp.write(block)
            esp.flush()
            sent += len(block)
            if sent < TOTAL_BYTES:
                reply = expect(esp, "ACK ")
                if reply != f"ACK {sent}":
                    raise RuntimeError(f"Unexpected acknowledgement: {reply}")

        out = expect(esp, "OUT ")
        result = expect(esp, "DONE ")
        elapsed = time.perf_counter() - start

    scores = [int(v) for v in out.split()[1:]]
    parts = result.split()
    if len(scores) != 16 or len(parts) != 8:
        raise RuntimeError(f"Malformed response: {out} / {result}")
    _, size, received_hash, winner, compute_us, validate_us, rx_us, device_us = parts
    if int(size) != TOTAL_BYTES or received_hash.lower() != expected_hash:
        raise RuntimeError("Transferred weight data failed hash/size check")
    if int(winner) != max(range(len(scores)), key=scores.__getitem__):
        raise RuntimeError("ESP32 output index disagrees with scores")

    print("\nPASS: weights verified; both layer kernels checked on ESP32")
    print("Output scores:", scores)
    print(f"Highest-scoring output index: {winner}")
    print(f"End-to-end: {elapsed * 1000:.3f} ms")
    print(f"Device elapsed: {int(device_us) / 1000:.3f} ms")
    print(f"USB receive/wait: {int(rx_us) / 1000:.3f} ms")
    print(f"INT4 compute: {int(compute_us) / 1000:.3f} ms")
    print(f"Reference validation: {int(validate_us) / 1000:.3f} ms")
    print("PHASE 6 TWO-LAYER FORWARD PASS COMPLETE")


if __name__ == "__main__":
    main()
