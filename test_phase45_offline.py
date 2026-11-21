"""Self-contained synthetic tests. These do NOT validate real Qwen output or hardware."""
import tempfile,json,struct,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import export_phase45_head as ex
from phase45_head_runner import head_reference

def main():
    with tempfile.TemporaryDirectory() as t:
        d=Path(t)
        # Make a tiny bona fide BF16-format safetensors fixture.
        def bf(v):return (np.asarray(v,dtype=np.float32).view(np.uint32)>>16).astype('<u2').tobytes()
        norm=[1.0]*896
        words=[[((i*17+j*13)%29-14)/24.0 for j in range(896)] for i in range(11)]
        blocks=[('model.norm.weight',norm,[896]),('lm_head.weight',sum(words,[]),[11,896])]
        hd={};blob=b''
        for name,values,shape in blocks:
            data=bf(values);start=len(blob);blob+=data;hd[name]={'dtype':'BF16','shape':shape,'data_offsets':[start,len(blob)]}
        header=json.dumps(hd).encode()
        fixture=d/'fake.safetensors';fixture.write_bytes(struct.pack('<Q',len(header))+header+blob)
        ex.VOCAB=11
        out=d/'head.bin'
        old=Path.cwd()
        import os
        try:
            os.chdir(d)
            ex.export(fixture,out)
        finally:os.chdir(old)
        meta=json.loads(out.with_suffix('.json').read_text())
        with open(out,'rb') as f:
            import mmap
            mm=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ)
            try:
                tok,score,runner=head_reference(mm,[0.1+((j*7)%13)/23.0 for j in range(896)],meta)
                assert 0<=tok<11 and np.isfinite(score)
            finally:mm.close()
        print('PHASE45 OFFLINE TEST PASS: safetensors parser, INT4 exporter, output-head reference',tok,round(score,5))
if __name__=='__main__':main()
