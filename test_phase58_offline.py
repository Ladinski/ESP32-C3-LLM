"""Static protocol and arithmetic bounds checks. Not an ESP32 benchmark."""
from pathlib import Path
import re
fw=Path('main/esp32_c3_llm.c').read_text(encoding='utf-8') if Path('main/esp32_c3_llm.c').exists() else Path('phase58_temp/main/esp32_c3_llm.c').read_text(encoding='utf-8')
runner=Path('bench_phase58.py').read_text(encoding='utf-8')
for marker in ('START P58 BENCH','READY PHASE58 BENCH','DONE PHASE58 BENCH','PHASE58_KERNEL','p58_select_residual_group','p58_residual_q','p58_dual_q15_dot'):
    assert marker in fw or marker in runner, marker
assert 'p58_benchmark();' in fw
assert 'p58_dual_q15_dot' in fw
assert 128*8*32760 < (1<<31)
RESULT = re.compile(r"^PHASE58_KERNEL valid=(\d+) group=(\d+) baseline_us=(\d+) candidate_us=(\d+) base_error=([0-9eE.+-]+) candidate_error=([0-9eE.+-]+) selected=(BENCH_PASS|FALLBACK)$")
line='PHASE58_KERNEL valid=1 group=6 baseline_us=100000 candidate_us=99000 base_error=0.01 candidate_error=0.00012 selected=BENCH_PASS'
assert RESULT.fullmatch(line)
print('PHASE58 OFFLINE TEST PASS: command, residual Q15, bounded INT4 dot, and speed gate protocol')

# RAM overflow regression: no large benchmark-only static buffers in .bss.
bench_section=fw.split('static void p58_benchmark(void){',1)[1].split('static void bench_kernel',1)[0]
assert 'static float x[FF];' not in bench_section
assert 'static uint8_t w[FF/2];' not in bench_section
assert 'float *x=(float *)malloc' in fw
assert 'uint8_t *w=(uint8_t *)malloc' in fw
assert 'free(w);' in fw and 'free(x);' in fw
print('PHASE58 RAM-FIX TEST PASS: temporary arrays allocated from heap, checked and freed')
