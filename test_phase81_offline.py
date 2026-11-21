import ast
from pathlib import Path
import numpy as np
from phase81_device_int8_audit import deterministic_input
p=Path(__file__).resolve().parent
source=(p/'main/esp32_c3_llm.c').read_text()
assert 'usb_serial_jtag_driver_install' in source
assert 'w[j]*x[j]' in source
assert '#define MAX_COLS 4864U' in source
assert np.array_equal(deterministic_input(896),deterministic_input(896))
assert deterministic_input(4864).shape == (4864,)
ast.parse((p/'phase81_device_int8_audit.py').read_text())
print('PHASE81 OFFLINE SYNTHETIC TEST PASS')
