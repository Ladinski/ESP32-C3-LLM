"""Resume Phase70 full-stack reference from already saved ESP32 capture; NO serial/ESP32 needed."""
import argparse,hashlib,json,mmap
from pathlib import Path
import numpy as np
from phase69_reference_math import reference
from phase70_two_token_verify import head_reference,verify_image

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--capture',default='phase70_device_capture.npz')
    p.add_argument('--image',default='phase25_qwen_24layers_int4.bin')
    p.add_argument('--metadata',default='phase25_qwen_24layers_int4.json')
    p.add_argument('--head',default='phase45_qwen_head_int4.bin')
    p.add_argument('--head-metadata',default='phase45_qwen_head_int4.json')
    p.add_argument('--atol',type=float,default=.001)
    p.add_argument('--rtol',type=float,default=.001)
    p.add_argument('--report',default='phase70_two_token_report.json')
    a=p.parse_args()
    if a.atol<=0 or a.rtol<0:raise ValueError('Invalid tolerance')
    with np.load(a.capture,allow_pickle=False) as d:
        device=d['device_final'].copy();positions=d['device_positions'].copy();embeddings=d['embeddings'].copy();ids=d['token_ids'].tolist()
        token=int(d['device_token']);device_logit=float(d['device_token_logit'])
    if device.shape!=(896,) or positions.shape!=(2,896) or embeddings.shape!=(len(ids),896) or len(ids)!=2:
        raise ValueError('Bad saved capture shapes')
    meta=json.loads(Path(a.metadata).read_text());hmeta=json.loads(Path(a.head_metadata).read_text())
    assert len(meta['layers'])==24
    print('MODEL VERIFIED',verify_image(a.image,meta),flush=True)
    print('HEAD VERIFIED',verify_image(a.head,hmeta),flush=True)
    with open(a.image,'rb') as f,open(a.head,'rb') as hf:
        mm=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ)
        hm=mmap.mmap(hf.fileno(),0,access=mmap.ACCESS_READ)
        try:
            expected=reference(mm,meta['layers'],list(embeddings))
            ref=np.asarray(expected[len(ids)-1,23],dtype=np.float32)
            for pos in (0,1):
                r=np.asarray(expected[pos,23],dtype=np.float64)
                delta=np.abs(positions[pos].astype(np.float64)-r)
                count=int(np.count_nonzero(delta>a.atol+a.rtol*np.abs(r)))
                print(f'PHASE70 POSITION {pos} FINAL HIDDEN: mixed_failures={count}/896 max_abs={delta.max():.9g}',flush=True)
            err=np.abs(device.astype(np.float64)-ref.astype(np.float64))
            fails=int(np.count_nonzero(err>a.atol+a.rtol*np.abs(ref.astype(np.float64))))
            print('PHASE70 FINAL HIDDEN mixed_failures=%d/896 max_abs=%.9g mean_abs=%.9g'%(fails,err.max(),err.mean()),flush=True)
            ref_token,ref_logit,runnerup=head_reference(hm,ref,hmeta)
            print('PHASE70 DEVICE TOKEN',token,'REFERENCE TOKEN',ref_token,'TOKEN MATCH',int(token==ref_token),flush=True)
            report=dict(device_token=token,reference_token=ref_token,token_match=token==ref_token,
                        hidden_mixed_failures=fails,hidden_max_abs=float(err.max()),hidden_mean_abs=float(err.mean()),
                        atol=a.atol,rtol=a.rtol,device_head_logit=device_logit,reference_head_logit=ref_logit,
                        reference_margin=ref_logit-runnerup,token_ids=ids,
                        comparison='Independent FP32 reference over genuine pretrained INT4 image; not original BF16 Qwen')
            Path(a.report).write_text(json.dumps(report,indent=2))
            print('REPORT',a.report,flush=True)
        finally: hm.close();mm.close()
if __name__=='__main__':main()
