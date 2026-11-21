import tempfile
from pathlib import Path
import numpy as np
from phase50_precision_audit import load,audit

with tempfile.TemporaryDirectory() as temp:
    path=Path(temp)/'trace.npz'
    one=np.ones(4864,dtype=np.float32)
    arrays={f'{side}_{stage}':one.copy() for stage in ('GATE','UP','ACT') for side in ('device','reference')}
    arrays['device_ACT'][813]=-1359.873
    arrays['reference_ACT'][813]=-1359.8975
    arrays['device_ACT'][2661]=-15.132532
    arrays['reference_ACT'][2661]=-15.130448
    np.savez_compressed(path,**arrays)
    assert audit(path,.001,.001,1.0)==0
    assert audit(path,.001,0,1.0)>=2
print('PHASE50 OFFLINE TEST PASS')
