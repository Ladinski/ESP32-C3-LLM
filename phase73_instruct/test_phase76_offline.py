"""Fast, synthetic, offline checks without model weights."""
import numpy as np
from phase76_selective_precision_gate import unpack_int4,quantize_int8,SCENARIOS

def test_unpack():
    import mmap, os, tempfile
    # Two signed nibbles: -8 and +7 => high nibble 7, low nibble 8.
    values=bytes([0x78])+np.array([0.5],dtype='<f4').tobytes()
    with tempfile.TemporaryFile() as f:
        f.write(values);f.flush()
        mm=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ)
        try:
            w=unpack_int4(mm,{'offset':0,'rows':1,'cols':2,'row_bytes':5})
            assert np.allclose(w,[[-4.,3.5]]),w
        finally:mm.close()

def test_int8():
    w=np.array([[0.,1.,-1.],[4.,-5.,0.]],dtype=np.float32)
    q=quantize_int8(w)
    assert q.shape==w.shape and np.max(np.abs(q-w))<0.04
    assert SCENARIOS['both_fp16'][(3,'down_proj')]=='fp16'

if __name__=='__main__':
    test_unpack();test_int8()
    print('PHASE76 OFFLINE SYNTHETIC TEST PASS')
