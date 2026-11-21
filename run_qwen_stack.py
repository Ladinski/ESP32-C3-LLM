"""Phase 37: speed-gated host streaming experiment based on Phase 36 firmware.

Phase 36: ESP32-C3 runs consecutive pretrained Qwen2.5-0.5B transformer layers.
The PC serves immutable weight blocks. NumPy computes references only after hardware finishes.
"""
import argparse, hashlib, json, math, mmap, time
from pathlib import Path
import numpy as np

D,KV,FF,PAGE=896,128,4864,512
FNV0,PRIME=2166136261,16777619

def check(image,metadata):
    image,metadata=Path(image),Path(metadata)
    meta=json.loads(metadata.read_text(encoding='utf-8'))
    if meta['model']!='Qwen/Qwen2.5-0.5B' or meta['image_size']!=image.stat().st_size:raise RuntimeError('Model metadata or image size mismatch')
    if len(meta['layers']) not in range(1,25) or meta['input_token_ids']!=[100,200] or meta['page']!=PAGE:raise RuntimeError('Unexpected metadata')
    if any(L['index']!=i or len(L['matrices'])!=7 for i,L in enumerate(meta['layers'])):raise RuntimeError('Layer layout mismatch')
    with image.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
    if sha!=meta['sha256']:raise RuntimeError('SHA256 mismatch')
    print(f'MODEL PASS: official Qwen2.5-0.5B {len(meta["layers"])}-layer INT4 image | {image.stat().st_size:,} bytes | SHA256 {sha}',flush=True)
    return meta

def accesses(meta):
    # Phase 29: load both input embeddings, then reuse each matrix weight row
    # across TWO prompt positions before progressing to the next layer.
    for pos in range(2):
        yield meta['embeddings'][pos],D*4
    for L in meta['layers']:
        for off,n in zip([L['norms'][0]]+L['biases'],[D*4,D*4,KV*4,KV*4]):yield off,n
        for m in L['matrices'][:4]:
            for r in range(m['rows']):yield m['offset']+r*m['row_bytes'],m['row_bytes']
        yield L['norms'][1],D*4
        for m in L['matrices'][4:]:
            for r in range(m['rows']):yield m['offset']+r*m['row_bytes'],m['row_bytes']

def expected(mm,meta,block_size=PAGE):
    prev=-1;fetches=hits=physical=records=0;fnv=FNV0
    for offset,n in accesses(meta):
        if offset<0 or offset+n>len(mm):raise RuntimeError('Invalid exported offsets')
        b=mm[offset:offset+n];records+=1
        for val in b:fnv=((fnv^val)*PRIME)&0xffffffff
        at=0
        while at<n:
            pos=offset+at;pg=pos//block_size*block_size
            if pg==prev:hits+=1
            else:prev=pg;fetches+=1;physical+=min(block_size,len(mm)-pg)
            at+=min(n-at,block_size-pos%block_size)
    return fetches,hits,physical,records,fnv

def streaming_schedule(meta,decode_mode=False):
    if decode_mode:
        for pos in range(2):
            yield ('PAGE',meta['embeddings'][pos],D*4)
            for layer in meta['layers']:
                for offset,n in zip([layer['norms'][0]]+layer['biases'],[D*4,D*4,KV*4,KV*4]):
                    yield ('PAGE',offset,n)
                for m in layer['matrices'][:4]:
                    yield ('RANGE',m['offset'],m['rows']*m['row_bytes'])
                yield ('PAGE',layer['norms'][1],D*4)
                for m in layer['matrices'][4:]:
                    yield ('RANGE',m['offset'],m['rows']*m['row_bytes'])
        return
    for pos in range(2):
        yield ('PAGE', meta['embeddings'][pos], D*4)
    for layer in meta['layers']:
        for offset,n in zip([layer['norms'][0]]+layer['biases'],[D*4,D*4,KV*4,KV*4]):
            yield ('PAGE',offset,n)
        for m in layer['matrices'][:4]:
            yield ('RANGE',m['offset'],m['rows']*m['row_bytes'])
        yield ('PAGE',layer['norms'][1],D*4)
        for m in layer['matrices'][4:]:
            yield ('RANGE',m['offset'],m['rows']*m['row_bytes'])

def expected_streaming(mm,meta,block_size,decode_mode=False):
    prev=-1;fetches=hits=physical=records=streams=stream_bytes=0;fnv=FNV0
    ordered_streams=[]
    for kind,offset,n in streaming_schedule(meta,decode_mode):
        if offset<0 or n<=0 or offset+n>len(mm):raise RuntimeError('Invalid stream schedule')
        if kind=='RANGE':
            streams+=1;stream_bytes+=n;physical+=n
            ordered_streams.append((offset,n))
        else:
            at=0
            while at<n:
                pos=offset+at;pg=pos//block_size*block_size
                if pg==prev:hits+=1
                else:prev=pg;fetches+=1;physical+=min(block_size,len(mm)-pg)
                at+=min(n-at,block_size-pos%block_size)
        if kind=='PAGE':
            data=mm[offset:offset+n]
            for val in data:fnv=((fnv^val)*PRIME)&0xffffffff
            records+=1
        else:
            # Streamed rows have the same FNV order as Phase 31's record reads.
            # Count records and hash in chunks to avoid unnecessary Python slicing.
            m=None
            for layer in meta['layers']:
                for candidate in layer['matrices']:
                    if candidate['offset']==offset:m=candidate;break
                if m is not None:break
            if m is None:raise RuntimeError('Matrix range not in metadata')
            width=m['row_bytes']
            for row in range(m['rows']):
                start=offset+row*width
                for val in mm[start:start+width]:fnv=((fnv^val)*PRIME)&0xffffffff
                records+=1
    return (fetches,hits,physical,records,fnv,streams,stream_bytes),ordered_streams

def vec(mm,offset,n):return np.frombuffer(mm[offset:offset+n*4],dtype='<f4').copy()
def matvec(mm,m,x):
    rows,cols=m['rows'],m['cols'];width=m['row_bytes'];result=np.empty(rows,dtype=np.float32)
    for start in range(0,rows,32):
        end=min(rows,start+32)
        block=np.frombuffer(mm[m['offset']+start*width:m['offset']+end*width],dtype=np.uint8).reshape(end-start,width)
        packed=block[:,:cols//2]
        lo=(packed&15).astype(np.int8);hi=(packed>>4).astype(np.int8)
        lo[lo>=8]-=16;hi[hi>=8]-=16
        values=np.empty((end-start,cols),dtype=np.float32)
        values[:,0::2]=lo;values[:,1::2]=hi
        scales=np.frombuffer(block[:,cols//2:].copy().tobytes(),dtype='<f4')
        result[start:end]=(values@x)*scales
    return result

def rms(x,g):return (x/np.sqrt(np.mean(x.astype(np.float64)**2)+1e-6)*g).astype(np.float32)
def rope(x,heads,pos):
    x=x.reshape(heads,64).copy()
    angle=pos/(1_000_000.**(np.arange(32)*2/64))
    a=x[:,:32].copy();b=x[:,32:].copy()
    x[:,:32]=a*np.cos(angle)-b*np.sin(angle)
    x[:,32:]=b*np.cos(angle)+a*np.sin(angle)
    return x.reshape(-1)

def reference(mm,meta):
    caches=[([],[]) for _ in meta['layers']];last=[];samples={}
    for pos in range(2):
        state=vec(mm,meta['embeddings'][pos],D)
        for i,L in enumerate(meta['layers']):
            m=L['matrices'];n1=vec(mm,L['norms'][0],D);n2=vec(mm,L['norms'][1],D)
            bq=vec(mm,L['biases'][0],D);bk=vec(mm,L['biases'][1],KV);bv=vec(mm,L['biases'][2],KV)
            x=rms(state,n1)
            q=rope(matvec(mm,m[0],x)+bq,14,pos).reshape(14,64)
            k=rope(matvec(mm,m[1],x)+bk,2,pos).reshape(2,64)
            v=(matvec(mm,m[2],x)+bv).reshape(2,64)
            keys,values=caches[i];keys.append(k);values.append(v)
            ctx=np.empty((14,64),dtype=np.float32)
            for h in range(14):
                idx=h//7
                scores=np.array([np.dot(q[h],keys[t][idx])/8 for t in range(pos+1)],dtype=np.float64)
                probs=np.exp(scores-scores.max());probs/=probs.sum()
                ctx[h]=sum((probs[t]*values[t][idx] for t in range(pos+1)))
            residual=state+matvec(mm,m[3],ctx.reshape(-1))
            norm=rms(residual,n2)
            gate=matvec(mm,m[4],norm);up=matvec(mm,m[5],norm)
            activated=gate/(1+np.exp(-np.clip(gate,-80,80)))*up
            state=residual+matvec(mm,m[6],activated.astype(np.float32))
            samples[pos,i]=state.copy()
            print(f'  NumPy reference token={pos}, layer={i+1}/{len(meta["layers"])}',flush=True)
        last.append(state)
    return last,samples

def run(image,meta,port,baud,timeout,storage_only,expected_block,active_layers,decode_mode=False,kernel_mode='grouped',stream_write_chunk=4096,max_layer_error=0.001,reference_decode_baseline=19.504):
    import serial
    from serial.serialutil import SerialException
    with open(image,'rb') as f:
        mm=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ)
        try:
            # Full bytewise FNV is costly: do it BEFORE opening COM3, otherwise the
            # device may time out waiting for the first requested block.
            selected=dict(meta)
            selected['layers']=meta['layers'][:active_layers]
            expect,stream_ranges=expected_streaming(mm,selected,expected_block,decode_mode)
            block_size=None
            range_index=0; streamed_bytes_host=0; host_range_write_seconds=0.0
            kernel=None; bench=None; prefill_mode=None; token_times={};layer_full={}
            print(f"PHASE 37 EXPERIMENT: write_chunk={stream_write_chunk} bytes; grouped INT4; active_layers={active_layers}; max_layer_error={max_layer_error}",flush=True)
            print(f"PRECHECK: block={expected_block}, expected PAGE reads={expect[0]:,}, matrix streams={expect[5]}, physical bytes={expect[2]:,}",flush=True)
            deadline=time.monotonic()+timeout
            dev=None;ready=False;sent=0;done=False;reqs=physical=0;stats=None;outs={};samples={};profile=None
            host_write_seconds=0.0;host_write_max=0.0
            try:
                while time.monotonic()<deadline and not done:
                    try:
                        if dev is None:
                            dev=serial.Serial(port,baudrate=baud,timeout=.3,write_timeout=5)
                            time.sleep(.7)
                        if not ready and time.monotonic()-sent>1.5:
                            dev.write(f'START {kernel_mode.upper()} {"DECODE " if decode_mode else ""}{active_layers}\n'.encode('ascii'));sent=time.monotonic()
                        raw=dev.readline()
                        if not raw:continue
                        line=raw.decode('ascii',errors='replace').strip()
                        if line=='READY PHASE36':ready=True;print(line,flush=True);continue
                        if not ready:continue
                        if line.startswith('PHASE38_KERNEL '):
                            print(line,flush=True)
                        elif line.startswith('KERNEL_BENCH '):
                            bench=line
                            print(line,flush=True)
                        elif line.startswith('PREFILL '):
                            prefill_mode=line.split(' ',1)[1]
                            if prefill_mode!=('TOKEN_MAJOR_DECODE' if decode_mode else 'LAYER_MAJOR_SHARED_WEIGHTS'):raise RuntimeError('Unexpected execution mode '+line)
                            print(line,flush=True)
                        elif line.startswith('KERNEL '):
                            kernel=line.split(' ',1)[1]
                            if kernel not in ('Q15_INT4','GROUPED_Q15_INT4','FP32_BASELINE'):raise RuntimeError('Unknown kernel '+line)
                            print(line,flush=True)
                        elif line.startswith('BLOCK_BYTES '):
                            block_size=int(line.split()[1]);
                            if block_size not in (512,2048,4096,8192,16384):raise RuntimeError('Unexpected firmware block size')
                            if block_size!=expected_block:raise RuntimeError(f'Firmware uses {block_size}B but --block-size={expected_block}B')
                            print(f'ESP32 block={block_size} bytes confirmed',flush=True)
                        elif line.startswith('RANGE '):
                            fields=line.split()
                            if len(fields)!=3:raise RuntimeError('Bad RANGE request '+line)
                            off,n=map(int,fields[1:])
                            if range_index>=len(stream_ranges) or (off,n)!=stream_ranges[range_index]:
                                raise RuntimeError(f'Invalid or out-of-order RANGE {off} {n}, index={range_index}')
                            if n<=0 or off<0 or off+n>len(mm):raise RuntimeError('RANGE outside image')
                            # One ESP32-authorized contiguous matrix transfer; chunked
                            # writes bound host USB staging, not device model memory.
                            for at in range(0,n,stream_write_chunk):
                                chunk=mm[off+at:off+min(at+stream_write_chunk,n)]
                                started=time.perf_counter()
                                written=dev.write(chunk)
                                dt=time.perf_counter()-started
                                host_range_write_seconds+=dt
                                host_write_max=max(host_write_max,dt)
                                if written!=len(chunk):raise RuntimeError('Short serial RANGE write')
                            streamed_bytes_host+=n
                            range_index+=1
                        elif line.startswith('PAGE '):
                            _,os,ns=line.split();off,n=int(os),int(ns)
                            if block_size is None:raise RuntimeError('PAGE before BLOCK_BYTES')
                            if off<0 or off>=len(mm) or off%block_size or n!=min(block_size,len(mm)-off):raise RuntimeError('Invalid storage request '+line)
                            t_write=time.perf_counter()
                            written=dev.write(mm[off:off+n])
                            duration=time.perf_counter()-t_write
                            host_write_seconds+=duration
                            host_write_max=max(host_write_max,duration)
                            if written!=n:raise RuntimeError(f'Short serial write: {written}/{n} bytes at offset {off}')
                            reqs+=1;physical+=n
                            if reqs%8192==0:print(f'  Served {reqs:,} model pages',flush=True)
                        elif line.startswith('LAYER_FULL '):
                            fields=line.split();pos,layer,start=map(int,fields[1:4]);values=[float(x) for x in fields[4:]]
                            if pos not in (0,1) or layer not in range(active_layers) or start not in range(0,D,16) or len(values)!=16 or (pos,layer,start) in layer_full or not all(map(math.isfinite,values)):
                                raise RuntimeError('Bad full layer diagnostics '+line)
                            layer_full[pos,layer,start]=values
                        elif line.startswith('LAYER '):
                            fields=line.split();pos,i=map(int,fields[1:3]);vals=np.array([float(x) for x in fields[3:]],dtype=np.float32)
                            if pos not in (0,1) or i not in range(active_layers) or vals.size!=8 or (pos,i) in samples or not np.all(np.isfinite(vals)):
                                raise RuntimeError('Bad per-layer sample '+line)
                            samples[pos,i]=vals
                            print(f'  ESP32 token={pos} layer={i+1}/{active_layers} complete',flush=True)
                        elif line.startswith('OUT '):
                            fields=line.split();pos,j=map(int,fields[1:3]);values=[float(x) for x in fields[3:]]
                            if pos not in (0,1) or j not in range(0,D,16) or len(values)!=16 or (pos,j) in outs or not all(map(math.isfinite,values)):
                                raise RuntimeError('Bad output chunk '+line)
                            outs[pos,j]=values
                        elif line.startswith('STATS '):
                            fields=line.split()
                            if len(fields)!=10:raise RuntimeError('Bad STATS '+line)
                            stats=tuple(map(int,fields[1:5]))+(int(fields[5],16),)+tuple(map(int,fields[6:]))
                            print(line,flush=True)
                        elif line.startswith('TOKEN_TIME '):
                            fields=line.split()
                            if len(fields)!=3:raise RuntimeError('Bad TOKEN_TIME')
                            pos,us=map(int,fields[1:]);
                            if not decode_mode or pos not in (0,1) or pos in token_times or us<=0:raise RuntimeError('Invalid token timing')
                            token_times[pos]=us
                            print(f'  {"Prefill" if pos==0 else "Single-token causal decode"}: {us/1e6:.3f}s',flush=True)
                        elif line.startswith('PROFILE '):
                            parts=line.split()
                            if len(parts)!=7:raise RuntimeError('Malformed PROFILE '+line)
                            profile=tuple(map(int,parts[1:]))
                            print(line,flush=True)
                        elif line.startswith('DONE PHASE36'):
                            print(line,flush=True);done=True
                        elif line.startswith(('CONFIG ','TOKEN ')):print(line,flush=True)
                        elif line.startswith('ERR '):raise RuntimeError('ESP32 firmware error: '+line)
                    except (SerialException,OSError) as e:
                        if ready:raise RuntimeError('USB disconnected while running inference') from e
                        if dev:
                            try:dev.close()
                            except Exception:pass
                        dev=None;time.sleep(.4)
                if not done:raise TimeoutError('Phase 34 timed out; verify COM port, flash and monitor closed')
                if block_size is None or kernel is None or bench is None or prefill_mode is None:raise RuntimeError('Missing block/kernel/benchmark announcement from firmware')
                expected_kernel='FP32_BASELINE' if kernel_mode=='fp32' else 'GROUPED_Q15_INT4'
                if kernel!=expected_kernel:raise RuntimeError(f'Kernel mismatch: wanted {expected_kernel}, device announced {kernel}')
                if stats is None or stats[:7]!=expect:raise RuntimeError(f'Storage mismatch: hardware={stats} expected={expect}')
                if (reqs,physical+streamed_bytes_host)!=(expect[0],expect[2]):raise RuntimeError('Physical model transfer count mismatch')
                if range_index!=len(stream_ranges) or streamed_bytes_host!=expect[6]:raise RuntimeError('Incomplete matrix streams')
                if stats[7]<32768 or stats[8]<256:raise RuntimeError('Heap/stack safety guard failed')
                if len(samples)!=2*active_layers or len(outs)!=2*D//16 or len(layer_full)!=2*active_layers*(D//16):raise RuntimeError('Missing layer sample/output chunks')
                if decode_mode and set(token_times)!={0,1}:raise RuntimeError('Missing per-token decode timings')
                print(f'STORAGE PASS: {reqs:,} ESP32 PAGE requests + {range_index} continuous matrix RANGE requests; {physical+streamed_bytes_host:,} physical bytes',flush=True)
                print(f'HOST SERIAL WRITE: PAGE={host_write_seconds:.3f}s, continuous matrices={host_range_write_seconds:.3f}s, max write={host_write_max*1000:.1f}ms',flush=True)
                if profile is None:raise RuntimeError('Missing on-device performance profile')
                req_us,rx_us,read_us,math_us,matrices,total_us=profile
                if total_us<=0 or rx_us+req_us>total_us or math_us>total_us or matrices!=active_layers*7*(2 if decode_mode else 1):raise RuntimeError('Inconsistent device timing counters')
                other_us=total_us-(req_us+rx_us+math_us)
                print('DEVICE PROFILE (on-device microsecond counters):',flush=True)
                for title,us in [('USB request sending',req_us),('Waiting for model bytes',rx_us),('INT4 row arithmetic',math_us),('All other work + instrumentation',other_us)]:
                    print(f'  {title}: {us/1e6:.3f}s ({100*us/total_us:.1f}%)',flush=True)
                print(f'  Row read inclusive time: {read_us/1e6:.3f}s (overlaps with storage times)',flush=True)
                print(f'  Total: {total_us/1e6:.3f}s | {matrices} weight matrices {"streamed per token" if decode_mode else "reused across 2 tokens"} | {reqs:,} pages + {range_index} streams | kernel={kernel}',flush=True)
                if not storage_only:
                    print('Computing independent NumPy reference AFTER hardware inference...',flush=True)
                    expected_output,expected_samples=reference(mm,selected)
                    print('LAYER-BY-LAYER FULL ACTIVATION DIAGNOSTICS (896 features each):',flush=True)
                    worst_layer_error=0.0
                    for key,sample in sorted(samples.items()):
                        ref=expected_samples[key]
                        actual_layer=np.array(sum((layer_full[key[0],key[1],j] for j in range(0,D,16)),[]),dtype=np.float32)
                        delta=np.abs(actual_layer-ref)
                        err=float(np.max(delta));avg=float(np.mean(delta));rms=float(np.sqrt(np.mean(delta.astype(np.float64)**2)))
                        worst_layer_error=max(worst_layer_error,err)
                        print(f'  token={key[0]} layer={key[1]+1}: max={err:.8g}, mean={avg:.8g}, rmse={rms:.8g}, >1e-3={int(np.count_nonzero(delta>1e-3))}, >1e-2={int(np.count_nonzero(delta>1e-2))}',flush=True)
                        if not np.allclose(actual_layer[:8],sample,rtol=1e-5,atol=1e-5):
                            raise RuntimeError('Full-layer output does not match 8-feature sample')
                    for pos in (0,1):
                        actual=np.array(sum((outs[pos,j] for j in range(0,D,16)),[]),dtype=np.float32)
                        expected_token=expected_output[pos]
                        err=np.max(np.abs(actual-expected_token))
                        print(f'Token {pos}: final {active_layers}-layer max absolute error {err:.7g} (acceptance rtol=.005, atol=.01)',flush=True)
                        if not np.allclose(actual,expected_token,rtol=.005,atol=.01):raise RuntimeError('Final output differs from INT4 NumPy reference')
                    if worst_layer_error > max_layer_error:
                        raise RuntimeError(f'PRECISION REGRESSION: worst per-layer error {worst_layer_error:.7g} > {max_layer_error}; DO NOT promote this variant')
                    print(f'PRECISION PASS: worst full-layer error {worst_layer_error:.7g} <= {max_layer_error}',flush=True)
                    if decode_mode and active_layers==2:
                        decode_seconds=token_times[1]/1e6
                        print(f'DECODE SPEED: {decode_seconds:.3f}s | reference {reference_decode_baseline:.3f}s',flush=True)
                        if decode_seconds >= reference_decode_baseline:
                            print('SPEED GATE: NO IMPROVEMENT; keep proven 4096-byte path',flush=True)
                        else:
                            print(f'SPEED GATE: CANDIDATE IMPROVEMENT {(reference_decode_baseline/decode_seconds-1)*100:.2f}%; repeat to confirm',flush=True)
                    print(f'PHASE 37 TEST PASS: {active_layers} pretrained layers, kernel={kernel_mode}, INT4; numerical guard enforced',flush=True)
                else:print(f'PHASE 36 STORAGE-ONLY PASS: {active_layers} pretrained layers, no PC neural inference',flush=True)
            finally:
                if dev:dev.close()
        finally:mm.close()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--layers',type=int,default=24)
    p.add_argument('--active-layers',type=int,default=2,help='Layers executed by device; use 2 before 24')
    p.add_argument('--image');p.add_argument('--metadata')
    p.add_argument('--port',default='COM3');p.add_argument('--baud',type=int,default=115200)
    p.add_argument('--timeout',type=float,default=10800)
    p.add_argument('--block-size',type=int,choices=(512,2048,4096,8192,16384),default=4096)
    p.add_argument('--check-only',action='store_true');p.add_argument('--storage-only',action='store_true')
    p.add_argument('--kernel',choices=('grouped','fp32'),default='grouped',help='Select runtime ESP32 arithmetic: grouped Q15 or original FP32; INT4 weights unchanged')
    p.add_argument('--stream-write-chunk',type=int,choices=(1024,2048,4096,8192,16384),default=4096,help='Host USB serial write granularity for continuous matrix RANGE transfers; firmware is unchanged')
    p.add_argument('--max-layer-error',type=float,default=0.001,help='Hard fail if any full 896-feature layer exceeds this error; default 0.001')
    p.add_argument('--reference-decode-baseline',type=float,default=19.504,help='Previously measured Phase 34 two-layer decode seconds for speed gate')
    p.add_argument('--decode-benchmark',action='store_true',help='Two fixed input positions, token-major KV-cache benchmark; NOT next-token generation')
    a=p.parse_args();stem=f'phase25_qwen_{a.layers}layers_int4'
    image=a.image or stem+'.bin';metadata=a.metadata or stem+'.json'
    meta=check(image,metadata)
    if len(meta['layers'])!=a.layers:raise RuntimeError('Runner layer count != exported model layer count')
    if not 1<=a.active_layers<=a.layers:raise RuntimeError('active-layers must be within exported layer count')
    if a.max_layer_error<=0 or a.reference_decode_baseline<=0:raise ValueError('Error/speed thresholds must be positive')
    if a.kernel=='fp32':raise ValueError('Phase 37 speed-first test intentionally excludes slow FP32 kernel; use previous Phase 36 runner only for separate debugging')
    if not a.check_only:run(image,meta,a.port,a.baud,a.timeout,a.storage_only,a.block_size,a.active_layers,a.decode_benchmark,a.kernel,a.stream_write_chunk,a.max_layer_error,a.reference_decode_baseline)
if __name__=='__main__':main()
