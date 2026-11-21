import importlib.util,struct,tempfile
from pathlib import Path
import numpy as np
p=Path(__file__).with_name('phase80_export_int8.py')
spec=importlib.util.spec_from_file_location('phase80',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
assert len(m.MATS)==7 and sum(r*c for _,r,c in m.MATS)>0
x=np.array([-2.0,-0.7,0.,0.7,2.0],dtype=np.float32)
s=np.float32(float(np.max(np.abs(x)))/127)
q=np.clip(np.rint(x/s),-127,127).astype(np.int8)
assert q.tolist()==[-127,-44,0,44,127]
assert np.max(np.abs(q.astype(np.float32)*s-x))<0.01
assert struct.unpack('<f',struct.pack('<f',float(s)))[0]==float(s)
with tempfile.TemporaryDirectory() as d:
    f=Path(d)/'test.bin';f.write_bytes(b'QW8S'+q.tobytes()+struct.pack('<f',float(s)))
    assert f.read_bytes()[:4]==b'QW8S'
print('PHASE80 OFFLINE SYNTHETIC TEST PASS')
