"""Phase 21 static memory-accounting and storage protocol smoke test.

Runs without hardware or a pretrained checkpoint. Does not verify inference.
"""
import re
from pathlib import Path
from run_pretrained_paged import requests_for, expected_pages

C = Path('main/esp32_c3_llm.c').read_text(encoding='utf-8')

def define(name):
    match = re.search(rf'^#define\s+{name}\s+(\d+)U?\b', C, flags=re.M)
    if not match: raise AssertionError(f'Missing firmware constant {name}')
    return int(match.group(1))

def main():
    layers = define('LAYERS')
    context = define('MAX_CONTEXT')
    max_tok = define('MAX_TOK')
    dim = define('DIM')
    ff = define('FF')
    page = define('PAGE_SIZE')
    vocab = define('VOCAB')
    assert (layers, context, max_tok, dim, ff, page, vocab) == (8,16,8,64,256,512,50257)
    # sizeof(float)=4; every persistent compute buffer is enumerated in C.
    kv = 2*layers*context*dim*4
    f32 = (7*max_tok*dim + 2*max_tok*ff + max_tok*dim)*4
    # arrays x,z,q,k,v,context,projected are 7, tmp+mlp are 2
    aux = 1024+page+512+(dim*3+ff)*4+5*4+5*4+max_tok*4
    total = kv+f32+aux
    assert total <= 160*1024, f'Persistent inference buffers exceed configured budget: {total}'
    assert 'STATIC_INFERENCE_BYTES' in C and '_Static_assert' in C
    assert 'heap_caps_get_minimum_free_size' in C
    assert 'uxTaskGetStackHighWaterMark' in C
    assert 'ERR MEMORY_BUDGET' in C
    for step, ids in ((1,[7454,2402,257,640]),(2,[7454,2402,257,640,11]),(3,[7454,2402,257,640,11,612])):
        records=requests_for(ids,step)
        pages,hits=expected_pages(records)
        assert len(pages) < 5000 and len(records)>50000 and hits>50000
        print(f'PASS storage sequence {step}: {len(records):,} logical rows, {len(pages):,} pages')
    print(f'PASS static-buffer estimate: {total:,} bytes; KV cache {kv:,} bytes')
    print('PASS telemetry fields and memory guard are in the firmware source')
    print('NOTE: actual ESP32 free heap, minimum heap, and stack watermark need hardware testing')

if __name__=='__main__':main()
