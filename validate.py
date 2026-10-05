"""Compare any implementation against the frozen fp64 oracle.
Metric: max |err| / max|out|  -- scale-relative, not element-relative.
Element-relative is meaningless when outputs pass near zero."""
import torch
from nsa_reference import nsa_forward

# Empirical floors from the reference itself at reduced precision.
# bf16 has ~8 mantissa bits (~3.9e-3 rel/op); three branches + weighted
# combination accumulate to ~1e-2 scaled. A kernel cannot beat these.
FLOOR = {torch.float32: 2e-6, torch.bfloat16: 3e-2}

def check(fn=None, dtype=torch.float32, path='/workspace/nsa/data/testset.pt', verbose=True):
    cases = torch.load(path)
    worst, worst_i = 0.0, -1
    for i, c in enumerate(cases):
        Q,K,V = c['Q'].cuda(), c['K'].cuda(), c['V'].cuda()
        o64 = c['out64'].cuda()
        if fn is None:   # self-test: reference at lower precision
            o,_ = nsa_forward(Q,K,V, **c['cfg'], dtype=dtype, sel_override=c['sel'])
        else:
            o = fn(Q.to(dtype), K.to(dtype), V.to(dtype), c['sel'].cuda(), c['cfg'])
        e = ((o.double()-o64).abs().max() / o64.abs().max()).item()
        if e > worst: worst, worst_i = e, i
    ok = worst <= FLOOR.get(dtype, 1e-6)
    if verbose:
        print(f"{len(cases)} cases | dtype={dtype} | worst scaled err {worst:.3e} "
              f"(case {worst_i}) | {'PASS' if ok else 'FAIL'}")
    return worst, ok

if __name__ == '__main__':
    check(dtype=torch.float32)
    check(dtype=torch.bfloat16)
