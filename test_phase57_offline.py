import numpy as np
from phase57_integer_residual_gate import q15, make_quantized, select_group, metrics, GROUP, NG

def test():
    z = np.zeros(GROUP, np.float32)
    q, s = q15(z)
    assert np.all(q == 0) and s == 1
    x = np.linspace(-3.1415, 5.165, NG*GROUP, dtype=np.float32)
    q, scales, residuals = make_quantized(x)
    assert q.shape == (NG,GROUP) and scales.shape == (NG,) and residuals.shape == (NG,GROUP)
    assert np.max(np.abs(x - (q.astype(np.float32)*scales[:,None]+residuals).reshape(-1))) < 1e-6
    g, scores = select_group(x, residuals)
    assert g == int(np.argmax(scores))
    # A second Q15 pass over residuals must reduce input reconstruction error.
    rq, step = q15(residuals[g])
    reconstructed = q[g].astype(np.float32)*scales[g] + rq.astype(np.float32)*step
    original = q[g].astype(np.float32)*scales[g]
    assert np.max(np.abs(x[g*GROUP:(g+1)*GROUP]-reconstructed)) <= np.max(np.abs(x[g*GROUP:(g+1)*GROUP]-original))
    mm = metrics(np.array([1.0005,5.0],np.float32), np.array([1.0,5.0],np.float32))
    assert mm['mixed_tolerance_failures'] == 0
    print('PHASE57 OFFLINE TEST PASS: signed residual-Q15, group selection, reconstruction and metrics')

if __name__ == '__main__':test()
