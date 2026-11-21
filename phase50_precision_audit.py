"""Phase 50: post-inference diagnostic of saved Phase 49 activation trace.

No hardware work, no weights, no model inference or token selection on PC.
The saved file contains only selected intermediate layer-3 activations.
"""
import argparse
from pathlib import Path
import numpy as np

FIELDS=('GATE','UP','ACT')

def load(path):
    path=Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Missing {path}. Phase 49 should have produced phase49_act_trace.npz in the project directory.')
    with np.load(path,allow_pickle=False) as data:
        expected={f'{prefix}_{field}' for prefix in ('device','reference') for field in FIELDS}
        missing=expected.difference(data.files)
        if missing: raise ValueError(f'Trace missing fields: {sorted(missing)}')
        arr={k:np.asarray(data[k],dtype=np.float64).copy() for k in expected}
    for name,a in arr.items():
        if a.shape!=(4864,) or not np.all(np.isfinite(a)):
            raise ValueError(f'Invalid {name}: shape={a.shape}; expected 4864 finite values')
    return arr

def characterize(name,actual,reference,abs_limit,rel_limit,rel_floor):
    diff=np.abs(actual-reference)
    # Combined absolute+relative tolerance: abs_err <= abs_limit + rel_limit*abs(ref)
    allowed=abs_limit+rel_limit*np.abs(reference)
    count=int(np.count_nonzero(diff>allowed))
    # Scale-aware diagnostic (do not interpret as a rigorous bound on later tokens).
    normalized=diff/np.maximum(np.abs(reference),rel_floor)
    imax=int(np.argmax(diff));nmax=int(np.argmax(normalized))
    print(f'{name}: absolute max={diff[imax]:.9g} at feature {imax}; mean={np.mean(diff):.9g}')
    print(f'  normalized |error|/max(|reference|,{rel_floor:g}): max={normalized[nmax]:.9g} at feature {nmax}, p99={np.percentile(normalized,99):.9g}')
    print(f'  fails combined abs={abs_limit:g} + rel={rel_limit:g}*|reference|: {count}/{len(diff)}')
    print(f'  worst absolute feature values: device={actual[imax]:.9g}, reference={reference[imax]:.9g}, allowance={allowed[imax]:.9g}')
    return count

def audit(path,abs_limit,rel_limit,rel_floor):
    data=load(path)
    print('PHASE 50 | OFFLINE SCALE-AWARE PRECISION AUDIT')
    print('Using previously saved Phase 49 trace. No COM3, firmware changes, model inference or PC token selection.')
    failed=0
    for name in FIELDS:
        failed+=characterize(name,data[f'device_{name}'],data[f'reference_{name}'],abs_limit,rel_limit,rel_floor)
    # SwiGLU amplification diagnostics: normalized effect at top-3 activation discrepancies
    dg,du,da=(data[f'device_{x}'] for x in FIELDS)
    rg,ru,ra=(data[f'reference_{x}'] for x in FIELDS)
    delta=np.abs(da-ra)
    print('Largest three ACT differences (diagnostic only):')
    for i in np.argsort(delta)[-3:][::-1]:
        i=int(i)
        print(f'  feature {i}: ACT abs={delta[i]:.9g}, relative={delta[i]/max(abs(ra[i]),1e-12):.9g}, '
              f'GATE abs={abs(dg[i]-rg[i]):.9g}, UP abs={abs(du[i]-ru[i]):.9g}')
    print(f'PHASE50 INTERMEDIATE COMBINED-TOLERANCE FAIL COUNT: {failed} across 3 x 4864 comparisons')
    print('IMPORTANT: A combined tolerance is a diagnostic, not proof of trustworthy full-model output.')
    print('The saved Phase 49 NPZ lacks layer hidden states and vocabulary logits. Full-model token agreement CANNOT be assessed from this file.')
    print('The unresolved layer-3 absolute error and possible downstream error accumulation remain. Do not claim full 24-layer accuracy.')
    return failed

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trace',default='phase49_act_trace.npz')
    p.add_argument('--absolute',type=float,default=0.001)
    p.add_argument('--relative',type=float,default=0.001)
    p.add_argument('--relative-floor',type=float,default=1.0)
    a=p.parse_args()
    if a.absolute<0 or a.relative<0 or a.relative_floor<=0:
        p.error('thresholds must be nonnegative, and relative floor positive')
    audit(a.trace,a.absolute,a.relative,a.relative_floor)

if __name__=='__main__':main()
