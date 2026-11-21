"""Phase 20 offline protocol/integrity test (no pretrained checkpoint or serial device needed)."""
from pathlib import Path
import tempfile
import numpy as np
from model_storage import TOTAL
from run_pretrained_paged import expected_pages, requests_for


def fnv_bytes(data, state=2166136261):
    for b in data:
        state = ((state ^ b) * 16777619) & 0xffffffff
    return state


def test_case(blob, ids, step):
    records = requests_for(ids,step)
    expected, hits = expected_pages(records)
    page = -1
    cached = b''
    fetched = []
    logical_hash = 2166136261
    logical_len = 0
    counted_hits = 0
    for off, length in records:
        result = bytearray()
        until = off+length
        while off < until:
            p = (off // 512) * 512
            if p == page:
                counted_hits += 1
            else:
                cached = blob[p:p+512]
                page = p
                fetched.append((p,len(cached)))
            n = min(until-off, len(cached)-(off-p))
            assert n > 0
            result.extend(cached[off-p:off-p+n])
            off += n
        assert len(result) == length
        logical_hash = fnv_bytes(result,logical_hash)
        logical_len += length
    assert fetched == expected
    assert counted_hits == hits
    assert logical_len == sum(n for _,n in records)
    assert sum(n for _,n in fetched) == sum(n for _,n in expected)
    print(f'TEST PASS step={step}: logical={len(records):,} records, physical={len(fetched):,} pages, bytes={logical_len:,}, FNV32={logical_hash:08x}, reused={hits:,}')


def main():
    # Deterministic fake storage bytes. This tests storage mechanics, NOT model accuracy.
    pattern=np.arange(0,256,dtype=np.uint8).tobytes()
    data=(pattern*((TOTAL+255)//256))[:TOTAL]
    assert len(data)==TOTAL
    ids=[7454,2402,257,640]
    test_case(data,ids,1)
    test_case(data,ids+[11],2)
    test_case(data,ids+[11,612],3)
    print('PHASE 20 OFFLINE STORAGE TEST PASS; hardware accuracy is not established by this test')

if __name__=='__main__':main()
