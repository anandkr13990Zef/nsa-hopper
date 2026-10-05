import torch, time
from torch.utils.cpp_extension import load

mod = load('nsa_v0', ['/workspace/nsa/src/nsa_v0.cu'],
           extra_cuda_cflags=['-arch=sm_90a','-O3','--use_fast_math'],
           extra_include_paths=['/workspace/nsa/cutlass/include'], verbose=False)

def ref_selected(Q,K,V,sel,l_prime,BQ,dtype=torch.float64):
    """Selected branch only, per-query-block selection. fp64 ground truth."""
    Q,K,V = Q.to(dtype),K.to(dtype),V.to(dtype)
    B,Hq,S,D = Q.shape; Hkv=K.shape[1]; G=Hq//Hkv; dev=Q.device
    NQB,KSEL = sel.shape[2], sel.shape[3]
    n_blk = (S+l_prime-1)//l_prime
    # padding (-1) routed to a scratch column so it cannot overwrite block 0
    keep = torch.zeros(B,Hkv,NQB,n_blk+1, dtype=torch.bool, device=dev)
    idx = torch.where(sel >= 0, sel, torch.full_like(sel, n_blk)).long()
    keep.scatter_(3, idx, True)
    keep = keep[..., :n_blk]
    tok = keep.repeat_interleave(l_prime,3)[:,:,:,:S]            # [B,Hkv,NQB,S]
    tok = tok.repeat_interleave(BQ,2)[:,:,:S,:]                  # [B,Hkv,S,S]
    causal = torch.tril(torch.ones(S,S,dtype=torch.bool,device=dev))
    mask = (tok & causal).repeat_interleave(G,1)
    s = (Q @ K.repeat_interleave(G,1).transpose(-1,-2)) / (D**0.5)
    s = s.masked_fill(~mask, float('-inf'))
    p = torch.nan_to_num(torch.softmax(s,-1), nan=0.0)
    return p @ V.repeat_interleave(G,1)

torch.manual_seed(7)
BQ, LP = 16, 64
for (B,Hq,Hkv,S,D,KSEL) in [(1,8,2,512,64,4),(1,8,2,512,64,8),(1,16,2,1024,128,8),
                            (1,4,4,512,64,4),(1,16,1,768,64,6),(1,8,2,2048,128,16)]:
    NQB = (S+BQ-1)//BQ; n_blk=(S+LP-1)//LP
    Q=torch.randn(B,Hq,S,D,device='cuda',dtype=torch.bfloat16)
    K=torch.randn(B,Hkv,S,D,device='cuda',dtype=torch.bfloat16)
    V=torch.randn(B,Hkv,S,D,device='cuda',dtype=torch.bfloat16)
    sel=torch.full((B,Hkv,NQB,KSEL),-1,dtype=torch.int32,device='cuda')
    for qb in range(NQB):                       # only causally-visible blocks
        avail = max(1,(qb*BQ)//LP + 1)
        k = min(KSEL, avail)
        perm = torch.randperm(avail)[:k]
        sel[:,:,qb,:k] = perm.to(torch.int32).cuda()
    cyc = torch.zeros(NQB*B*Hkv*2, dtype=torch.int64, device='cuda')
    o = mod.forward(Q,K,V,sel,LP,BQ,cyc)
    r = ref_selected(Q,K,V,sel,LP,BQ)
    err = ((o.double()-r).abs().max()/r.abs().max()).item()
    w,mm = cyc[0::2].float().mean().item(), cyc[1::2].float().mean().item()
    print(f"B{B} Hq{Hq} Hkv{Hkv} S{S} D{D} k{KSEL}: err {err:.3e} "
          f"{'PASS' if err<3e-2 else 'FAIL'}  load/math cycles {w:.2e}/{mm:.2e} "
          f"({100*w/(w+mm):.1f}% load)")
