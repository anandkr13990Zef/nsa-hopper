import torch, statistics
from torch.utils.cpp_extension import load
from ref_only import ref_selected
flags=dict(extra_cuda_cflags=['-arch=sm_90a','-O3','--use_fast_math'],
           extra_include_paths=['/workspace/nsa/cutlass/include'],verbose=False)
v1=load('nsa_v1',['/workspace/nsa/src/nsa_v1.cu'],**flags)
v2=load('nsa_v2',['/workspace/nsa/src/nsa_v2.cu'],**flags)

def bench(mod,Q,K,V,sel,LP,BQ,cyc,it=60):
    for _ in range(15): mod.forward(Q,K,V,sel,LP,BQ,cyc)
    torch.cuda.synchronize(); ts=[]
    for _ in range(it):
        a=torch.cuda.Event(True); b=torch.cuda.Event(True)
        a.record(); mod.forward(Q,K,V,sel,LP,BQ,cyc); b.record()
        torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    return statistics.median(ts), min(ts)

torch.manual_seed(7); LP=64
print(f"{'config':<34}{'v1 ms':>9}{'v2 ms':>9}{'speedup':>9}{'v2 wait%':>10}")
for (B,Hq,Hkv,S,D,KSEL,BQ) in [(1,8,2,4096,64,8,16),(1,16,2,8192,128,16,8),
                               (1,8,2,16384,128,16,16),(1,4,4,8192,64,16,64)]:
    G=Hq//Hkv; NQB=(S+BQ-1)//BQ
    Q=torch.randn(B,Hq,S,D,device='cuda',dtype=torch.bfloat16)
    K=torch.randn(B,Hkv,S,D,device='cuda',dtype=torch.bfloat16)
    V=torch.randn(B,Hkv,S,D,device='cuda',dtype=torch.bfloat16)
    sel=torch.full((B,Hkv,NQB,KSEL),-1,dtype=torch.int32,device='cuda')
    for qb in range(NQB):
        av=max(1,(qb*BQ)//LP+1); k=min(KSEL,av)
        sel[:,:,qb,:k]=torch.randperm(av)[:k].to(torch.int32).cuda()
    cyc=torch.zeros(NQB*B*Hkv*2,dtype=torch.int64,device='cuda')
    m1,_=bench(v1,Q,K,V,sel,LP,BQ,cyc)
    m2,_=bench(v2,Q,K,V,sel,LP,BQ,cyc)
    w=cyc[0::2].float().mean().item(); mm=cyc[1::2].float().mean().item()
    print(f"Hq{Hq} Hkv{Hkv} S{S} D{D} k{KSEL}".ljust(34)+
          f"{m1:9.3f}{m2:9.3f}{m1/m2:8.2f}x{100*w/(w+mm):9.1f}%")
