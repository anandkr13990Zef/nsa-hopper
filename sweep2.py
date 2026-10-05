"""Crossover sweep. First sweep showed: LP and KSEL dominate, G is flat
(BQ*G==64 coupling), and v2 LOSES to v1 at D=128/LP=64/k=4 (0.87x).
This resolves that boundary: LP {32,64,128,256} x KSEL {1,2,4,8,16,32}.
"""
import torch, itertools, csv, os, statistics, subprocess, time
from torch.utils.cpp_extension import load
F=dict(extra_cuda_cflags=['-arch=sm_90a','-O3','--use_fast_math'],
       extra_include_paths=['/workspace/nsa/cutlass/include'],verbose=False)
v1=load('nsa_v1',['/workspace/nsa/src/nsa_v1.cu'],**F)
v2=load('nsa_v2',['/workspace/nsa/src/nsa_v2.cu'],**F)

def clk():
    try: return int(subprocess.check_output(
        "nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits",
        shell=True).decode().split('\n')[0])
    except Exception: return -1

def timed(fn,warm=20,it=60):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); ts=[]
    for _ in range(it):
        a=torch.cuda.Event(True); b=torch.cuda.Event(True)
        a.record(); fn(); b.record(); torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return statistics.median(ts),min(ts),max(ts)

rows=[]; t0w=time.time()
grid=list(itertools.product([(4,16),(16,4)],[64,128],[32,64,128,256],
                            [4096,8192,16384,32768],[1,2,4,8,16,32]))
print(f"{len(grid)} configs")
print(f"{'G':>3}{'D':>5}{'LP':>5}{'S':>7}{'k':>4}{'v1ms':>9}{'v2ms':>9}"
      f"{'spd':>7}{'wait%':>7}{'B/stage':>9}{'TF/s':>7}")
torch.manual_seed(7)
for (G,BQ),D,LP,S,KSEL in grid:
    if S%LP: continue
    Hkv=2; Hq=Hkv*G; B=1; NQB=(S+BQ-1)//BQ
    n_blk=S//LP
    if KSEL>n_blk: continue
    try:
        Q=torch.randn(B,Hq,S,D,device='cuda',dtype=torch.bfloat16)
        K=torch.randn(B,Hkv,S,D,device='cuda',dtype=torch.bfloat16)
        V=torch.randn(B,Hkv,S,D,device='cuda',dtype=torch.bfloat16)
        sel=torch.full((B,Hkv,NQB,KSEL),-1,dtype=torch.int32,device='cuda')
        for qb in range(NQB):
            av=max(1,(qb*BQ)//LP+1); k=min(KSEL,av)
            sel[:,:,qb,:k]=torch.randperm(av)[:k].to(torch.int32).cuda()
        cyc=torch.zeros(NQB*B*Hkv*2,dtype=torch.int64,device='cuda')
        m1,l1,_=timed(lambda: v1.forward(Q,K,V,sel,LP,BQ,cyc))
        m2,l2,h2=timed(lambda: v2.forward(Q,K,V,sel,LP,BQ,cyc))
        w=cyc[0::2].float().mean().item(); mm=cyc[1::2].float().mean().item()
        nsel=(sel>=0).float().sum().item()/(B*Hkv*NQB)
        flops=2*2*B*Hq*NQB*BQ*nsel*LP*D
        bps=2*2*LP*D                      # bytes per staged KV block
        rows.append(dict(G=G,BQ=BQ,D=D,LP=LP,S=S,KSEL=KSEL,nsel_avg=round(nsel,2),
            v1_med=round(m1,4),v1_min=round(l1,4),v2_med=round(m2,4),v2_min=round(l2,4),
            speedup=round(m1/m2,3),wait_pct=round(100*w/(w+mm),2),
            bytes_per_stage=bps,stages_in_flight=round(min(nsel,4),2),
            tflops=round(flops/(l2*1e-3)/1e12,1),
            gbs=round(2*2*B*Hkv*NQB*nsel*LP*D/(l2*1e-3)/1e9,1),
            spread_pct=round(100*(h2-l2)/m2,1),sm_clk=clk()))
        r=rows[-1]
        print(f"{G:>3}{D:>5}{LP:>5}{S:>7}{KSEL:>4}{m1:9.3f}{m2:9.3f}"
              f"{r['speedup']:6.2f}x{r['wait_pct']:7.1f}{bps:9d}{r['tflops']:7.1f}")
        del Q,K,V,sel,cyc
    except RuntimeError as e:
        print(f"{G:>3}{D:>5}{LP:>5}{S:>7}{KSEL:>4}   SKIP {type(e).__name__}")
    torch.cuda.empty_cache()

out='/workspace/nsa/results/sweep2.csv'
os.makedirs(os.path.dirname(out),exist_ok=True)
with open(out,'w',newline='') as f:
    w_=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w_.writeheader(); w_.writerows(rows)
print(f"\n{len(rows)} rows -> {out}  ({(time.time()-t0w)/60:.1f} min)")

# crossover summary
print("\ncrossover (speedup < 1.0):")
for r in rows:
    if r['speedup']<1.0:
        print(f"  G{r['G']} D{r['D']} LP{r['LP']} S{r['S']} k{r['KSEL']}"
              f" -> {r['speedup']}x  ({r['bytes_per_stage']}B/stage)")
