
from pathlib import Path
import time
import serial

PORT = "COM3"
BAUD = 115200

BLOCK_SIZE = 512
SMALL_SIZE = 8192
LARGE_SIZE = 262144

SOURCE_FILE = Path("test_weights_int4.bin")
LARGE_FILE = Path("stream_weights_int4.bin")


def prepare_weights():
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(
            f"Missing {SOURCE_FILE}. "
            "Keep the weight file from Phase 5.2."
        )

    original = SOURCE_FILE.read_bytes()

    if len(original) != SMALL_SIZE:
        raise ValueError(
            f"Expected {SMALL_SIZE} bytes, "
            f"found {len(original)}"
        )

    # Repeat the test data to simulate a larger
    # PC-resident weight matrix.
    LARGE_FILE.write_bytes(
        original * (LARGE_SIZE // SMALL_SIZE)
    )

    return original


def read_response(esp, expected):
    deadline = time.monotonic() + 15

    while time.monotonic() < deadline:
        raw = esp.readline()

        if not raw:
            continue

        text = raw.decode(
            "ascii", errors="replace"
        ).strip()

        if not text:
            continue

        if text.startswith("ERR"):
            raise RuntimeError(text)

        if text.startswith(expected):
            return text

    raise TimeoutError(
        f"Timeout waiting for {expected}"
    )


def send_command(esp, command):
    esp.write((command + "\n").encode("ascii"))
    esp.flush()


def test_stream(esp, path, size):
    print()
    print(f"--- Testing {size:,} bytes ---")

    send_command(esp, f"START {size}")
    response = read_response(esp, "GO ")

    if response != f"GO {size}":
        raise RuntimeError(response)

    started = time.perf_counter()

    with path.open("rb") as file:
        bytes_sent = 0

        while bytes_sent < size:
            block = file.read(BLOCK_SIZE)

            if len(block) != BLOCK_SIZE:
                raise RuntimeError("Unexpected end of file")

            esp.write(block)
            esp.flush()

            bytes_sent += len(block)

            if bytes_sent == size:
                response = read_response(esp, "DONE ")
            else:
                response = read_response(
                    esp, f"ACK {bytes_sent}"
                )

                if response != f"ACK {bytes_sent}":
                    raise RuntimeError(response)

    elapsed = time.perf_counter() - started

    parts = response.split()

    if len(parts) != 6:
        raise RuntimeError(
            f"Unexpected response: {response}"
        )

    received = int(parts[1])
    checksum = int(parts[2])
    compute_us = int(parts[3])
    validation_us = int(parts[4])
    device_elapsed_us = int(parts[5])

    if received != size:
        raise RuntimeError(
            f"Received {received}, expected {size}"
        )

    macs = (size * 2)

    # Every packed byte has two INT4 weights,
    # each contributing one MAC.
    compute_macs_per_sec = (
        macs * 1_000_000 / compute_us
        if compute_us else 0
    )

    print("Status: PASS")
    print(f"Checksum: {checksum}")
    print(f"Bytes transferred: {received:,}")
    print(f"End-to-end time: {elapsed:.4f} s")
    print(
        f"Device elapsed time: "
        f"{device_elapsed_us / 1000:.2f} ms"
    )
    print(
        f"ESP32 optimized compute: "
        f"{compute_us / 1000:.3f} ms"
    )
    print(
        f"ESP32 validation: "
        f"{validation_us / 1000:.3f} ms"
    )
    print(
        f"Effective throughput: "
        f"{received / elapsed:,.2f} bytes/s"
    )
    print(
        f"Compute throughput: "
        f"{compute_macs_per_sec:,.2f} MAC/s"
    )


def main():
    original = prepare_weights()

    print("ESP32-C3 Binary INT4 Benchmark")
    print(f"COM port: {PORT}")
    print(f"Block size: {BLOCK_SIZE} bytes")
    print(f"Small test: {len(original):,} bytes")
    print(f"Large test: {LARGE_SIZE:,} bytes")

    with serial.Serial(
        PORT,
        BAUD,
        timeout=0.25,
        write_timeout=10,
    ) as esp:
        time.sleep(1)
        esp.reset_input_buffer()

        send_command(esp, "HELLO")
        print(read_response(esp, "READY"))

        test_stream(
            esp,
            SOURCE_FILE,
            SMALL_SIZE
        )

        test_stream(
            esp,
            LARGE_FILE,
            LARGE_SIZE
        )

    print("\nALL STREAMING TESTS PASSED")


if __name__ == "__main__":
    main()
