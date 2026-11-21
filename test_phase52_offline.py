"""Synthetic tests of Phase52 INT4 signed nibble and linear attribution."""
import io
import json
import mmap
import os
import tempfile
import numpy as np
from phase52_mlp_error_attribution import projection_from_int4


def main():
    rows, cols = 896, 4864
    perrow = cols // 2 + 4
    # Build a constant matrix with weights alternating [+1,-1] and scale 0.25.
    matrixbytes = bytearray()
    for _ in range(rows):
        matrixbytes.extend(bytes([0xf1]) * (cols // 2))
        matrixbytes.extend(np.float32(0.25).tobytes())
    with tempfile.TemporaryFile() as f:
        f.write(matrixbytes);f.flush()
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            delta = np.zeros(cols, dtype=np.float32)
            delta[813] = -0.0244140625
            delta[2661] = 0.00208377838
            p = projection_from_int4(mm,(0,rows,cols,perrow),np.stack([delta]))[0]
            expected = sum((0.25 if j%2==0 else -0.25)*float(delta[j]) for j in (813,2661))
            assert np.allclose(p, expected, atol=1e-6), (p[:2], expected)
    print('PHASE52 OFFLINE TEST PASS: signed INT4 unpack + per-row scales + linear projection')

if __name__ == '__main__':
    main()
