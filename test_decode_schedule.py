"""Small synthetic layout validates Phase 33's token-major storage pattern."""
from pathlib import Path
import tempfile,mmap
import run_qwen_stack as r

def align(n): return (n+511)//512*512
with tempfile.TemporaryDirectory() as d:
    cursor=0
    def alloc(n):
        global cursor
        off=cursor;cursor+=align(n);return off
    embeddings=[alloc(r.D*4) for _ in range(2)]
    layers=[]
    for i in range(2):
        norms=[alloc(r.D*4) for _ in range(2)]
        biases=[alloc(r.D*4),alloc(r.KV*4),alloc(r.KV*4)]
        matrices=[]
        for rows,cols in zip((896,128,128,896,4864,4864,896),(896,896,896,896,896,896,4864)):
            width=cols//2+4
            matrices.append({'offset':alloc(rows*width),'rows':rows,'cols':cols,'row_bytes':width})
        layers.append({'norms':norms,'biases':biases,'matrices':matrices})
    path=Path(d)/'sparse_model.bin'
    with path.open('wb') as f:f.truncate(cursor)
    with path.open('rb') as f,mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as mm:
        meta={'embeddings':embeddings,'layers':layers}
        pre,pre_ranges=r.expected_streaming(mm,meta,4096,False)
        dec,dec_ranges=r.expected_streaming(mm,meta,4096,True)
        assert len(pre_ranges)==14 and len(dec_ranges)==28
        assert dec_ranges[:7]==pre_ranges[:7]
        assert dec_ranges[14:21]==pre_ranges[:7]
        assert dec[5]==28 and dec[6]==pre[6]*2
        assert dec[3]==2*(pre[3]-2)+2  # embeddings read once each
        assert dec[2]>pre[2] and dec[2]<2.1*pre[2]
        print('PHASE 33 DECODE STORAGE SCHEDULE PASS')
        print(f'Prefill: {pre[0]} pages, {pre[5]} streams, {pre[2]:,} physical bytes')
        print(f'Token-major: {dec[0]} pages, {dec[5]} streams, {dec[2]:,} physical bytes')
