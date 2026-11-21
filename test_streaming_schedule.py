"""Offline Phase 32 two-layer storage scheduling validation; no pretrained weights needed."""
import mmap, tempfile
from pathlib import Path
import run_qwen_stack as r

def aligned(n):return ((n+511)//512)*512
with tempfile.TemporaryDirectory() as d:
    path=Path(d)/'dummy.bin'
    cursor=0
    def alloc(n):
        global cursor
        pos=cursor;cursor+=aligned(n);return pos
    embeddings=[alloc(r.D*4) for _ in range(2)]
    layers=[]
    for layer in range(2):
        norms=[alloc(r.D*4) for _ in range(2)]
        biases=[alloc(r.D*4),alloc(r.KV*4),alloc(r.KV*4)]
        matrices=[]
        for rows,cols in zip((896,128,128,896,4864,4864,896),(896,896,896,896,896,896,4864)):
            width=cols//2+4
            matrices.append({'offset':alloc(rows*width),'rows':rows,'cols':cols,'row_bytes':width})
        layers.append({'norms':norms,'biases':biases,'matrices':matrices})
    with path.open('wb') as f:f.truncate(cursor)
    with path.open('rb') as f:
        mm=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ)
        meta={'embeddings':embeddings,'layers':layers}
        legacy=r.expected(mm,meta,4096)
        stream, ranges=r.expected_streaming(mm,meta,4096)
        assert len(ranges)==14 and len(set(ranges))==14
        assert stream[3:5]==legacy[3:5]
        assert stream[5]==14 and stream[6]==sum(n for _,n in ranges)
        assert stream[0]<40 and stream[2]>=stream[6]
        assert stream[2] < legacy[2]
        assert stream[3]==25356
        print('PHASE 32 STORAGE SCHEDULE PASS')
        print('Original 4KB PAGE transfers:',legacy[0],legacy[2])
        print('Phase32 PAGE transfers:',stream[0],'matrix streams:',stream[5], 'total bytes:',stream[2])
        mm.close()
