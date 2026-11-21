"""Phase 44: read-only analysis of Qwen INT4 autoregressive weight traffic.
No serial, no flashing, no inference, no model modifications.
"""
import argparse
import json
from pathlib import Path

MIB = 1024 * 1024

def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', default='phase25_qwen_24layers_int4.json')
    parser.add_argument('--mib-s', type=float, default=0.771, help='Measured Phase 43 median USB throughput')
    parser.add_argument('--free-heap', type=int, default=93720, help='Measured ESP32 free heap bytes')
    parser.add_argument('--cache-bytes', type=int, default=32768, help='Optimistic dedicated RAM for cached immutable weights')
    parser.add_argument('--summary-json', default='phase44_decode_feasibility.json')
    a = parser.parse_args()
    if a.mib_s <= 0 or a.free_heap < 0 or a.cache_bytes < 0:
        parser.error('throughput must be positive; memory budgets nonnegative')
    meta = json.loads(Path(a.metadata).read_text(encoding='utf-8'))
    layers = meta.get('layers', [])
    if len(layers) != 24: raise ValueError('This study requires the real 24-layer metadata')
    if meta.get('model') != 'Qwen/Qwen2.5-0.5B': raise ValueError('Unexpected model')
    layer_bytes=[]
    matrix_bytes=[]
    other_bytes=[]
    for idx, layer in enumerate(layers):
        if layer['index'] != idx or len(layer['matrices']) != 7: raise ValueError('Malformed layer layout')
        mats = sum(int(m['rows']) * int(m['row_bytes']) for m in layer['matrices'])
        norms = 2 * 896 * 4
        biases = (896 + 128 + 128) * 4
        layer_bytes.append(mats + norms + biases)
        matrix_bytes.append(mats)
        other_bytes.append(norms + biases)
    matrix_total=sum(matrix_bytes)
    all_weight_bytes=sum(layer_bytes)
    biggest_matrix=max(int(m['rows'])*int(m['row_bytes']) for layer in layers for m in layer['matrices'])
    smallest_layer=min(layer_bytes)
    cache=min(a.cache_bytes, a.free_heap)
    # An optimistic upper bound: even if every cache byte could replace a distinct
    # necessary weight byte on every decode, no more than cache bytes can be saved.
    optimistic_saving=min(cache,all_weight_bytes)
    bound_seconds=(all_weight_bytes-optimistic_saving)/(a.mib_s*MIB)
    result={
        'model': meta['model'], 'layers':len(layers),
        'matrix_bytes_per_decode':matrix_total,
        'norm_bias_bytes_per_decode':sum(other_bytes),
        'total_required_layer_bytes_per_decode':all_weight_bytes,
        'image_bytes':meta.get('image_size'),
        'smallest_layer_bytes':smallest_layer,
        'largest_matrix_bytes':biggest_matrix,
        'usb_mib_per_second':a.mib_s,
        'ideal_usb_transfer_seconds':all_weight_bytes/(a.mib_s*MIB),
        'free_heap_bytes':a.free_heap,
        'optimistic_cache_bytes':cache,
        'optimistic_cache_fraction_percent':100*optimistic_saving/all_weight_bytes,
        'optimistic_cached_transfer_seconds':bound_seconds,
        'note':'Traffic-only estimate, not prediction of actual decode latency; ignores model head and computational work.'
    }
    print('PHASE 44 | REAL QWEN INT4 DECODE FEASIBILITY | NO HARDWARE CHANGES')
    print('Model:',meta['model'],'| layers:',len(layers))
    print(f'Matrix data per decoded token: {matrix_total:,} bytes ({matrix_total/MIB:.2f} MiB)')
    print(f'Norms and biases per decoded token: {sum(other_bytes):,} bytes')
    print(f'All layer data per decoded token: {all_weight_bytes:,} bytes ({all_weight_bytes/MIB:.2f} MiB)')
    print(f'Smallest pretrained layer: {smallest_layer/MIB:.2f} MiB; largest matrix: {biggest_matrix/MIB:.2f} MiB')
    print(f'USB Phase 43 median throughput: {a.mib_s:.3f} MiB/s')
    print(f'Idealized transfer-only latency per full 24-layer decode: {result["ideal_usb_transfer_seconds"]:.1f} s')
    print(f'Optimistic dedicated weight cache: {cache:,} bytes, saves <= {result["optimistic_cache_fraction_percent"]:.3f}% of traffic')
    print(f'Estimated transfer-only latency with that cache at measured throughput: {bound_seconds:.1f} s')
    print('CAUTION: estimates assume sustained benchmark USB rate and exclude protocol overhead; compute can overlap with transfers. Missing final norm, vocab logits and sampling.')
    print('VERDICT: Layer scheduling and small SRAM weight caches cannot provide a major autoregressive decode speedup for this 24-layer model.')
    print('NEXT ENGINEERING MILESTONE: real on-device last norm + output head + sampled token, before further microoptimizations.')
    Path(a.summary_json).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print('REPORT:',a.summary_json)

if __name__ == '__main__':run()
