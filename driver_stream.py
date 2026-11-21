from pathlib import Path
import time
import serial

PORT = "COM3"
BLOCK_SIZE = 512
SMALL_SIZE = 8192
LARGE_SIZE = 262144
SOURCE_FILE = Path("test_weights_int4.bin")
LARGE_FILE = Path("stream_weights_int4.bin")
FNV_OFFSET = 2166136261
FNV_PRIME = 16777619


def prepare_weights():
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(f"Missing {SOURCE_FILE} from Phase 5.2")
    small = SOURCE_FILE.read_bytes()
    if len(small) != SMALL_SIZE:
        raise ValueError(f"Expected {SMALL_SIZE} bytes, got {len(small)}")
    if not LARGE_FILE.exists() or LARGE_FILE.stat().st_size != LARGE_SIZE:
        LARGE_FILE.write_bytes(small * (LARGE_SIZE // SMALL_SIZE))


def file_hash(path):
    h = FNV_OFFSET
    with path.open("rb") as f:
        while chunk := f.read(8192):
            for byte in chunk:
                h = ((h ^ byte) * FNV_PRIME) & 0xFFFFFFFF
    return h


def get_reply(esp, prefix, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        data = esp.readline()
        if not data:
            continue
        line = data.decode("ascii", errors="replace").strip()
        if line.startswith("ERR "):
            raise RuntimeError(f"ESP32: {line}")
        if line.startswith(prefix):
            return line
    raise TimeoutError(f"No reply beginning {prefix!r}")


def send_command(esp, message):
    esp.write((message + "\n").encode("ascii"))
    esp.flush()


def test_stream(esp, path, expected_size):
    print(f"\n--- {expected_size:,} bytes ---")
    send_command(esp, f"START {expected_size}")
    response = get_reply(esp, "GO ")
    if response != f"GO {expected_size}":
        raise RuntimeError(f"Unexpected GO: {response}")

    start = time.perf_counter()
    sent = 0
    with path.open("rb") as f:
        while sent < expected_size:
            chunk = f.read(BLOCK_SIZE)
            if len(chunk) != BLOCK_SIZE:
                raise ValueError("Unexpected end of file")
            esp.write(chunk)
            esp.flush()
            sent += len(chunk)
            if sent < expected_size:
                actual = get_reply(esp, "ACK ")
                if actual != f"ACK {sent}":
                    raise RuntimeError(f"Unexpected ACK: {actual}")
            else:
                result = get_reply(esp, "DONE ")

    elapsed = time.perf_counter() - start
    parts = result.split()
    if len(parts) != 8:
        raise ValueError(f"Bad DONE response: {result}")
    _, received, checksum, compute_us, validate_us, receive_us, device_us, hash_hex = parts
    received, checksum = int(received), int(checksum)
    compute_us, validate_us = int(compute_us), int(validate_us)
    receive_us, device_us = int(receive_us), int(device_us)
    device_hash = int(hash_hex, 16)
    expected_hash = file_hash(path)

    if received != expected_size or device_hash != expected_hash:
        raise RuntimeError(
            f"Transfer mismatch: bytes={received}/{expected_size}, "
            f"hash={device_hash:08x}/{expected_hash:08x}"
        )

    print("PASS: full transfer and byte hash verified")
    print(f"ESP32 checksum: {checksum}")
    print(f"Weight hash: {device_hash:08x}")
    print(f"End-to-end: {elapsed:.4f} s")
    print(f"ESP32 elapsed: {device_us / 1000:.3f} ms")
    print(f"Time waiting/reading: {receive_us / 1000:.3f} ms")
    print(f"INT4 compute: {compute_us / 1000:.3f} ms")
    print(f"Validation: {validate_us / 1000:.3f} ms")
    print(f"Effective throughput: {received / elapsed:,.2f} bytes/s")
    print(f"Compute throughput: {2 * received * 1e6 / compute_us:,.2f} MAC/s")


def main():
    prepare_weights()
    print("ESP32-C3 Phase 5.4 | USB driver benchmark")
    print(f"Port: {PORT}, block: {BLOCK_SIZE} bytes")
    with serial.Serial(PORT, 115200, timeout=0.25, write_timeout=10) as esp:
        time.sleep(1)
        esp.reset_input_buffer()
        send_command(esp, "HELLO")
        print(get_reply(esp, "READY"))
        test_stream(esp, SOURCE_FILE, SMALL_SIZE)
        test_stream(esp, LARGE_FILE, LARGE_SIZE)
    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
