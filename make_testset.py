"""Frozen (Q,K,V,selection,output) tuples with fp64 ground truth.
Every kernel from here on validates against this file, not against itself."""
import torch, itertools
from nsa_reference import nsa_forward

torch.manual_seed(1337)
cases, meta = [], []
grid = [
    # (B, Hq, Hkv, S, D, n_sel, l_prime, window)   -- note the edge cases
    (1, 8,  2, 512,  64,  4, 64, 128),
    (1, 8,  2, 512,  64, 16, 64, 128),   # n_sel > available blocks at low positions
    (2, 16, 2, 1024, 128, 8, 64, 256),
    (1, 4,  4, 512,  64,  4, 64, 128),   # group size 1 (no GQA sharing)
    (1, 16, 1, 768,  64,  8, 128, 256),  # group size 16
    (1, 8,  2, 500,  64,  4, 64, 128),   # S not divisible by l_prime
    (1, 8,  2, 512,  64,  1, 256, 64),   # minimal selection
    (1, 8,  2, 2048, 128, 16, 64, 512),
]
for i, (B,Hq,Hkv,S,D,n_sel,lp,w) in enumerate(grid):
    for rep in range(25):                       # 8 shapes x 25 = 200 tuples
        Q = torch.randn(B,Hq,S,D, device='cuda')
        K = torch.randn(B,Hkv,S,D, device='cuda')
        V = torch.randn(B,Hkv,S,D, device='cuda')
        if rep == 0:                            # adversarial: large-magnitude row
            Q[..., 0, :] *= 50.0
        o64, sel = nsa_forward(Q,K,V, n_sel=n_sel, l_prime=lp, window=w,
                               dtype=torch.float64)
        cases.append(dict(Q=Q.cpu(), K=K.cpu(), V=V.cpu(),
                          sel=sel.cpu(), out64=o64.cpu(),
                          cfg=dict(n_sel=n_sel, l_prime=lp, window=w)))
    meta.append((B,Hq,Hkv,S,D,n_sel,lp,w))
    print(f"shape {i}: {(B,Hq,Hkv,S,D)} done")

torch.save(cases, '/workspace/nsa/data/testset.pt')
print(f"saved {len(cases)} tuples")

# fp32 vs fp64 gap — the floor below which no kernel error is meaningful
c = cases[0]
o32,_ = nsa_forward(c['Q'].cuda(), c['K'].cuda(), c['V'].cuda(),
                    **c['cfg'], dtype=torch.float32)
err = ((o32.double()-c['out64'].cuda()).abs()/(c['out64'].cuda().abs()+1e-9)).max()
print(f"fp32-vs-fp64 max rel err: {err.item():.3e}   <- your kernel cannot beat this")
