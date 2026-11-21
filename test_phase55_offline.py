import numpy as np
import mmap
import tempfile
from pathlib import Path
from phase55_selective_correction import quantize,rank_groups,read_row_chunk,COLS,GROUP,GROUPS

def main():
    assert COLS==4864 and GROUPS==38
    x=np.zeros(COLS,dtype=np.float32)
    q,s,r=quantize(x)
    assert np.all(q==0) and np.all(s==1) and np.all(r==0)
    x=np.sin(np.arange(COLS,dtype=np.float32)*0.019).astype(np.float32)
    q,s,r=quantize(x)
    assert q.shape==(38,128) and s.shape==(38,)
    reconstructed=(q.astype(np.float32)*s[:,None]+r).reshape(-1)
    assert np.max(np.abs(reconstructed-x))<1e-6
    for method in ('residual_l1','residual_rms','residual_max','activation_max'):
        order,_=rank_groups(x,r,method)
        assert sorted(order.tolist())==list(range(GROUPS))
    # Regression: the old code used (end-start) instead of end in the
    # absolute end offset, making batch 2 (rows 16:32) an empty slice.
    row_bytes=COLS//2+4
    prefix=19
    rows=48
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/'synthetic.bin'
        data=bytearray(prefix+rows*row_bytes)
        for row in range(rows):
            data[prefix+row*row_bytes]=row
        path.write_bytes(data)
        with path.open('rb') as file, mmap.mmap(file.fileno(),0,access=mmap.ACCESS_READ) as mm:
            for start,end in ((0,16),(16,32),(32,48)):
                chunk=read_row_chunk(mm,prefix,start,end,row_bytes)
                assert chunk.shape==(16,row_bytes)
                assert np.array_equal(chunk[:,0],np.arange(start,end,dtype=np.uint8))
    print('PHASE55 OFFLINE TEST PASS: exact group ranking + multi-batch row offsets')

if __name__=='__main__':main()
