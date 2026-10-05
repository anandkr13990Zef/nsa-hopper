import torch
from torch.utils.cpp_extension import load
from ref_only import ref_selected
mod = load('nsa_v2',['/workspace/nsa/src/nsa_v2.cu'],
  extra_cuda_cflags=['-arch=sm_90a','-O3','--use_fast_math','--expt-relaxed-constexpr'],
  extra_include_paths=['/workspace/nsa/cutlass/include'], verbose=True)
torch.manual_seed(7); LP=64
# v1 constraint: BQ*G == 64
for (B,Hq,Hkv,S,D,KSEL,BQ) in [(1,8,2,512,64,4,16),(1,16,2,1024,128,8,8),
                               (1,4,4,512,64,4,64),(1,16,1,1024,64,6,4),
                               (1,8,2,2048,128,16,16)]:
    G=Hq//Hkv; assert BQ*G==64, (BQ,G)
    NQB=(S+BQ-1)//BQ
    Q=torch.randn(B,Hq,S,D,device='cuda',dtype=torch.bfloat16)
    K=torch.randn(B,Hkv,S,D,device='cuda',dtype=torch.bfloat16)
    V=torch.randn(B,Hkv,S,D,device='cuda',dtype=torch.bfloat16)
    sel=torch.full((B,Hkv,NQB,KSEL),-1,dtype=torch.int32,device='cuda')
    for qb in range(NQB):
        av=max(1,(qb*BQ)//LP+1); k=min(KSEL,av)
        sel[:,:,qb,:k]=torch.randperm(av)[:k].to(torch.int32).cuda()
    cyc=torch.zeros(NQB*B*Hkv*2,dtype=torch.int64,device='cuda')
    o=mod.forward(Q,K,V,sel,LP,BQ,cyc); r=ref_selected(Q,K,V,sel,LP,BQ)
    e=((o.double()-r).abs().max()/r.abs().max()).item()
    w,mm=cyc[0::2].float().mean().item(),cyc[1::2].float().mean().item()
    print(f"Hq{Hq} Hkv{Hkv} S{S} D{D} k{KSEL} BQ{BQ}: err {e:.3e} "
          f"{'PASS' if e<3e-2 else 'FAIL'}  {100*w/(w+mm):.1f}% load")
