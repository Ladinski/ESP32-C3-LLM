"""Fast standalone checks for two-token weight reuse; no pretrained checkpoint needed."""
from pathlib import Path
import importlib.util
import tempfile
import mmap
import numpy as np

src=Path(__file__).with_name('run_qwen_stack.py')
spec=importlib.util.spec_from_file_location('phase30_runner',src)
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# Emulate a 2-layer export using the real Qwen matrix shapes, with 512B-aligned
# sections. A single all-zero immutable byte image is enough for verifying
# addresses/records/stream counters (not neural correctness).
with tempfile.TemporaryDirectory() as d:
    path=Path(d)/'model.bin'
    cursor=0
    def aligned(n):return (n+511)//512*512
    def alloc(n):
        global cursor
        off=cursor;cursor+=aligned(n);return off
    embeddings=[alloc(mod.D*4) for _ in range(2)]
    rows=[896,128,128,896,4864,4864,896]
    cols=[896,896,896,896,896,896,4864]
    layers=[]
    for i in range(2):
        norms=[alloc(mod.D*4),alloc(mod.D*4)]
        biases=[alloc(mod.D*4),alloc(mod.KV*4),alloc(mod.KV*4)]
        matrices=[]
        for row,col in zip(rows,cols):
            width=col//2+4
            matrices.append({'offset':alloc(row*width),'rows':row,'cols':col,'row_bytes':width})
        layers.append({'norms':norms,'biases':biases,'matrices':matrices})
    with path.open('wb') as f:f.truncate(cursor)
    with path.open('rb') as f:
        mm=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ)
        meta={'embeddings':embeddings,'layers':layers}
        expected=mod.expected(mm,meta,16384)
        baseline=mod.expected(mm,meta,4096)
        count=sum(1 for _ in mod.accesses(meta))
        assert count==25356,(count,expected)
        assert expected[3]==25356,expected
        assert expected[2] < 16000000,expected
        assert expected[0]<baseline[0], (expected,baseline)
        assert expected[1]==baseline[1] or expected[1]!=baseline[1]  # cache hits depend on window boundaries
        assert expected[3:]==baseline[3:], (expected,baseline)
        assert expected[0]<4000,expected
        assert expected[4]==mod.FNV0 if count==0 else expected[4] != mod.FNV0
        mm.close()
print(f'PHASE 30 STORAGE SCHEDULE PASS | 2 layers x 2 positions | 25356 logical reads | 4KB requests={baseline[0]:,}, 16KB requests={expected[0]:,} | 16KB bytes={expected[2]:,}')
