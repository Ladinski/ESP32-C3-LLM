from pathlib import Path
import time
import serial

PORT = "COM3"
BLOCK_SIZE = 512
SOURCE_FILE = Path("test_weights_int4.bin")
FNV_OFFSET = 2166136261
FNV_PRIME = 16777619
TEST_SIZES = (262_144, 4 * 1024 * 1024, 16 * 1024 * 1024)


def make_test_file(source: bytes, size: int) -> Path:
    path = Path(f"weights_{size // 1048576}MiB.bin") if size >= 1048576 else Path("weights_256KiB.bin")
    if path.exists() and path.stat().st_size == size:
        return path
    with path.open("wb") as f:
        for _ in range(size // len(source)):
            f.write(source)
    return path


def fnv32_update(h: int, buf: bytes) -> int:
    for byte in buf:
        h = ((h ^ byte) * FNV_PRIME) & 0xFFFFFFFF
    return h


def get_reply(esp, expected: str, timeout: float = 15.0) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        raw = esp.readline()
        if not raw:
            continue
        line = raw.decode("ascii", errors="replace").strip()
        if line.startswith("ERR "):
            raise RuntimeError(f"ESP32 error: {line}")
        if line.startswith(expected):
            return line
    raise TimeoutError(f"No response beginning {expected!r}")


def send_command(esp, command: str) -> None:
    esp.write((command + "\n").encode("ascii"))
    esp.flush()


def test_stream(esp, path: Path, size: int, source_checksum: int) -> None:
    print(f"\n--- Streaming {size:,} bytes ({size / 1048576:.2f} MiB) ---", flush=True)
    send_command(esp, f"START {size}")
    response = get_reply(esp, "GO ")
    if response != f"GO {size}":
        raise RuntimeError(f"Unexpected GO: {response}")

    started = time.perf_counter()
    sent, host_hash = 0, FNV_OFFSET
    next_progress = 1024 * 1024
    with path.open("rb") as f:
        while sent < size:
            block = f.read(BLOCK_SIZE)
            if len(block) != BLOCK_SIZE:
                raise RuntimeError("Unexpected end of weight file")
            host_hash = fnv32_update(host_hash, block)
            written = esp.write(block)
            if written != BLOCK_SIZE:
                raise RuntimeError(f"Short host write: {written}")
            esp.flush()
            sent += BLOCK_SIZE

            if sent == size:
                result = get_reply(esp, "DONE ")
            else:
                ack = get_reply(esp, "ACK ")
                if ack != f"ACK {sent}":
                    raise RuntimeError(f"Incorrect ACK at {sent}: {ack}")

            if sent >= next_progress:
                print(f"  Sent {sent / 1048576:.0f} MiB", flush=True)
                next_progress += 1024 * 1024

    elapsed = time.perf_counter() - started
    parts = result.split()
    if len(parts) != 8:
        raise RuntimeError(f"Unexpected DONE response: {result}")
    _, received_str, checksum_str, compute_str, validation_str, receive_str, device_str, hash_str = parts
    received = int(received_str)
    checksum = int(checksum_str)
    compute_us = int(compute_str)
    validation_us = int(validation_str)
    receive_us = int(receive_str)
    device_us = int(device_str)
    device_hash = int(hash_str, 16)
    expected_checksum = source_checksum * (size // len(SOURCE_FILE.read_bytes()))

    if received != size or device_hash != host_hash or checksum != expected_checksum:
        raise RuntimeError(
            f"Verification failed: bytes {received}/{size}, "
            f"hash {device_hash:08x}/{host_hash:08x}, "
            f"checksum {checksum}/{expected_checksum}"
        )

    print("PASS: byte hash and INT4 checksum match")
    print(f"Bytes: {received:,} | Hash: {device_hash:08x} | Checksum: {checksum}")
    print(f"End-to-end: {elapsed:.3f} s | ESP32 elapsed: {device_us / 1e6:.3f} s")
    print(f"ESP32 receive/wait: {receive_us / 1e6:.3f} s")
    print(f"INT4 compute: {compute_us / 1000:.3f} ms | Validation: {validation_us / 1000:.3f} ms")
    print(f"Effective throughput: {received / elapsed:,.0f} bytes/s")
    print(f"INT4 compute throughput: {2 * received * 1e6 / compute_us:,.0f} MAC/s")


def main() -> None:
    if not SOURCE_FILE.is_file():
        raise FileNotFoundError("Missing test_weights_int4.bin from Phase 5.2")
    source = SOURCE_FILE.read_bytes()
    if len(source) != 8192:
        raise ValueError(f"Expected 8192-byte source, found {len(source)}")
    # The deterministic Phase 5.2 file produced checksum 4793 on this firmware.
    # Repeating it N times should produce exactly 4793*N.
    SOURCE_CHECKSUM = 4793
    files = [(size, make_test_file(source, size)) for size in TEST_SIZES]

    print("ESP32-C3 Phase 5.5 | Multi-megabyte INT4 streaming")
    print("PC sends weights; ESP32 computes all dot products")
    with serial.Serial(PORT, 115200, timeout=0.25, write_timeout=10) as esp:
        time.sleep(1)
        esp.reset_input_buffer()
        send_command(esp, "HELLO")
        print(get_reply(esp, "READY"))
        for size, path in files:
            test_stream(esp, path, size, SOURCE_CHECKSUM)
    print("\nALL PHASE 5.5 TESTS PASSED")


if __name__ == "__main__":
    main()
