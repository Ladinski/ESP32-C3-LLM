
from pathlib import Path
import time
import serial

PORT = "COM3"
ROWS = 64
COLS = 256
CHUNK_BYTES = 16
WEIGHT_FILE = Path("test_weights_int4.bin")

BYTES_PER_ROW = COLS // 2
CHUNKS_PER_ROW = BYTES_PER_ROW // CHUNK_BYTES
EXPECTED_SIZE = ROWS * BYTES_PER_ROW


def prepare_weights():
    if WEIGHT_FILE.exists():
        data = WEIGHT_FILE.read_bytes()

        if len(data) != EXPECTED_SIZE:
            raise ValueError(
                f"Weight file must be {EXPECTED_SIZE} bytes"
            )

        return data

    # Prepare deterministic synthetic packed INT4 weights.
    # This is data generation, not model inference.
    state = 0x12345678
    data = bytearray()

    for _ in range(EXPECTED_SIZE):
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= state >> 17
        state ^= (state << 5) & 0xFFFFFFFF
        state &= 0xFFFFFFFF

        data.append(state & 0xFF)

    WEIGHT_FILE.write_bytes(data)
    return bytes(data)


def read_response(esp, expected_prefix):
    deadline = time.monotonic() + 10

    while time.monotonic() < deadline:
        raw = esp.readline()

        if not raw:
            continue

        message = raw.decode(
            "utf-8", errors="replace"
        ).strip()

        if not message:
            continue

        if message.startswith("ERR"):
            raise RuntimeError(f"ESP32 error: {message}")

        if message.startswith(expected_prefix):
            return message

    raise TimeoutError(
        f"Timed out waiting for {expected_prefix}"
    )


def send_line(esp, message):
    esp.write((message + "\n").encode("ascii"))
    esp.flush()


def main():
    weights = prepare_weights()

    print("ESP32-C3 INT4 Streaming Test")
    print(f"Weight file: {WEIGHT_FILE}")
    print(f"Total weight bytes: {len(weights)}")
    print(f"Matrix: {ROWS} x {COLS}")
    print(f"Chunk size: {CHUNK_BYTES} bytes")

    with serial.Serial(
        PORT,
        115200,
        timeout=0.25,
        write_timeout=5,
    ) as esp:
        time.sleep(1)
        esp.reset_input_buffer()

        send_line(esp, "HELLO")
        print(read_response(esp, "READY"))

        send_line(esp, "START")
        print(read_response(esp, "GO "))

        started = time.perf_counter()

        for row in range(ROWS):
            for index in range(CHUNKS_PER_ROW):
                offset = (
                    row * BYTES_PER_ROW
                    + index * CHUNK_BYTES
                )

                piece = weights[
                    offset:offset + CHUNK_BYTES
                ]

                command = (
                    f"B{row:02X}{index:X}:"
                    f"{piece.hex().upper()}"
                )

                send_line(esp, command)

                if index == CHUNKS_PER_ROW - 1:
                    response = read_response(
                        esp, f"ROW {row} "
                    )
                    print(response)
                else:
                    read_response(
                        esp, f"ACK {row} {index}"
                    )

        result = read_response(esp, "DONE ")
        elapsed = time.perf_counter() - started

    parts = result.split()

    if len(parts) != 4:
        raise ValueError(f"Unexpected result: {result}")

    checksum = int(parts[1])
    received = int(parts[2])
    compute_us = int(parts[3])

    if received != EXPECTED_SIZE:
        raise RuntimeError(
            f"Only {received}/{EXPECTED_SIZE} bytes received"
        )

    print("\n========== RESULTS ==========")
    print("Status: PASS")
    print(f"ESP32 checksum: {checksum}")
    print(f"Weight bytes received: {received}")
    print(f"End-to-end time: {elapsed:.4f} s")
    print(f"ESP32 compute + validation: {compute_us} us")
    print(
        f"Effective transfer rate: "
        f"{received / elapsed:.2f} bytes/s"
    )
    print("=============================")


if __name__ == "__main__":
    main()
