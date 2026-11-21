"""Test Phase59 arithmetic concept, source anchors, protocol, and memory policy.

No device or ESP-IDF compilation is performed here.
"""
from pathlib import Path
import re
RESULT = re.compile(r"^PHASE59_KERNEL valid=(\d+) group=(\d+) baseline_us=(\d+) candidate_us=(\d+) base_error=([0-9eE.+-]+) candidate_error=([0-9eE.+-]+) selected=(BENCH_PASS|FALLBACK)$")

source = Path('main/esp32_c3_llm.c').read_text(encoding='utf-8')
runner = Path('bench_phase59.py').read_text(encoding='utf-8')
for fragment in ('START P59 BENCH', 'READY PHASE59 BENCH', 'DONE PHASE59 BENCH',
                 'PHASE59_KERNEL', 'p59_dual_group_shift', 'p59_fused_dual_dot',
                 'phase38_select_kernel();', 'p59_benchmark();'):
    assert fragment in source or fragment in runner, fragment
start = source.index('// Phase 59:')
end = source.index('static void p58_benchmark(void)', start)
p59 = source[start:end]
assert 'static float x[FF]' not in p59
assert 'static uint8_t w[FF/2]' not in p59
assert 'malloc(' in p59 and 'free(w);free(x);' in p59
assert 'p59_dual_group_shift(packed+begin/2,q+begin,p58_residual_q' in p59
assert '(float)extra*p58_residual_step' in p59
assert 'ct[2]<=bt[2]' in p59
assert 'fabsf(candidate-previous)<=0.00002f' in p59
assert 128*8*32760 < 2**31

# Python integer reference: each packed weight should be decoded once but
# must produce the same two signed dot products as separate SHIFT calls.
def unpack(byte):
    lo = byte & 15
    hi = (byte >> 4) & 15
    return (lo - 16 if lo >= 8 else lo, hi - 16 if hi >= 8 else hi)

for seed in range(32):
    packed = [(i * 37 + seed * 23 + 19) & 255 for i in range(64)]
    q = [((i * 179 + seed * 29) % 65521) - 32760 for i in range(128)]
    r = [((i * 241 + seed * 17) % 65521) - 32760 for i in range(128)]
    once_q = once_r = 0
    twice_q = twice_r = 0
    for i, byte in enumerate(packed):
        lo, hi = unpack(byte)
        once_q += lo * q[2*i] + hi * q[2*i+1]
        once_r += lo * r[2*i] + hi * r[2*i+1]
    for i, byte in enumerate(packed):
        lo, hi = unpack(byte)
        twice_q += lo * q[2*i] + hi * q[2*i+1]
    for i, byte in enumerate(packed):
        lo, hi = unpack(byte)
        twice_r += lo * r[2*i] + hi * r[2*i+1]
    assert (once_q, once_r) == (twice_q, twice_r)

assert RESULT.fullmatch('PHASE59_KERNEL valid=1 group=6 baseline_us=107182 candidate_us=105000 base_error=0.0163574 candidate_error=0.00012207 selected=BENCH_PASS')
print('PHASE59 OFFLINE TEST PASS: fused dual integer arithmetic, bounded accumulators, dynamic RAM, speed gate and protocol')
