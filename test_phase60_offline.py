import numpy as np
from phase60_outlier_quantization_gate import q15, quantize_all, choose_group, create_candidates, unpack_chunk, GROUP, COLS

def main():
    x=np.zeros(COLS,np.float32)
    x[6*GROUP:7*GROUP]=np.linspace(-2,2,GROUP,dtype=np.float32)
    x[6*GROUP+45]=1200
    q, steps=quantize_all(x)
    selected,_=choose_group(x,q,steps)
    assert selected==6,(selected,steps[6])
    candidates,order=create_candidates(x,q,steps,selected)
    assert order[0]==45
    v=x[6*GROUP:7*GROUP]
    w=np.arange(GROUP,dtype=np.int32)%16-8
    exact=np.float64(w)@v.astype(np.float64)
    baseline=(w@q[6].astype(np.int32))*float(steps[6])
    peel=next(c for c in candidates if c['name']=='peel_top_1')
    assert peel['indices'].tolist()==[45]
    peeled=(w@peel['q'].astype(np.int32))*float(peel['step']) + w[45]*float(peel['values'][0])
    assert abs(peeled-exact)<abs(baseline-exact), (baseline,peeled,exact)
    assert np.all(peel['q'][peel['indices']]==0)
    # Test packing, signed INT4 decoding, and absolute offsets for batch 16:32.
    rows=32; rb=COLS//2+4; off=13
    a=np.zeros((rows,rb),np.uint8)
    a[16:,0]=0x8F
    a[:,COLS//2:COLS//2+4]=np.array([0,0,128,63],np.uint8)
    data=bytes(off)+a.tobytes()
    ww,scales=unpack_chunk(data,off,16,32,rb)
    assert ww.shape==(16,COLS) and ww[0,0]==-1 and ww[0,1]==-8
    assert np.all(scales==1)
    assert len(create_candidates(x,q,steps,selected)[0])==7
    print('PHASE60 OFFLINE TEST PASS: outlier-only scaling, sparse correction, INT4 batch offsets, candidate selection')

if __name__=='__main__':main()
