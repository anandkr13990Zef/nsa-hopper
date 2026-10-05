"""Phase-diagram sweep. v1 (wgmma, sync load) vs v2 (wgmma + multistage
cp.async) vs dense cuDNN SDPA. One row per config.

Clocks are unlockable on this host, so: 20 warmup + 60 timed, median AND min
reported, SM clock sampled per config. Treat min as the least-throttled
estimate; spread column flags configs where throttling makes the number soft.
"""
import torch, itertools, csv, os, statistics, subprocess, time
from torch.utils.cpp_extension import load
from torch.nn.attention import SDPBackend, sdpa_kernel

F = dict(extra_cuda_cflags=['-arch=sm_90a','-O3','--use_fast_math'],
         extra_include_paths=['/workspace/nsa/cutlass/include'], verbose=False)
v1 = load('nsa_v1', ['/workspace/nsa/src/nsa_v1.cu'], **F)
v2 = load('nsa_v2', ['/workspace/nsa/src/nsa_v2.cu'], **F)

def clk():
    try: return int(subprocess.check_output(
        "nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits",
        shell=True).decode().split('\n')[0])
    except Exception: return -1

def timed(fn, warm=20, it=60):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); ts=[]; cl=[]
    for _ in range(it):
        a=torch.cuda.Event(True); b=torch.cuda.Event(True)
        a.record(); fn(); b.record(); torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    cl.append(clk())
    return statistics.median(ts), min(ts), max(ts), cl[0]

def make_sel(B,Hkv,NQB,KSEL,BQ,LP,S):
    sel=torch.full((B,Hkv,NQB,KSEL),-1,dtype=torch.int32,device='cuda')
    for qb in range(NQB):
        av=max(1,(qb*BQ)//LP+1); k=min(KSEL,av)
        sel[:,:,qb,:k]=torch.randperm(av)[:k].to(torch.int32).cuda()
    return sel

rows=[]; t_start=time.time()
G_BQ = [(1,64),(2,32),(4,16),(8,8),(16,4)]
grid = list(itertools.product(G_BQ,[64,128],[64,128],[4096,8192,16384],[4,8,16,32]))
print(f"{len(grid)} configs\n")
print(f"{'G':>3}{'BQ':>4}{'D':>5}{'LP':>5}{'S':>7}{'k':>4}"
      f"{'v1ms':>9}{'v2ms':>9}{'spd':>7}{'wait%':>7}{'AI':>8}{'TF/s':>8}")

torch.manual_seed(7)
for (G,BQ),D,LP,S,KSEL in grid:
    Hkv=2; Hq=Hkv*G; B=1
    if S % LP: continue
    NQB=(S+BQ-1)//BQ
    try:
        Q=torch.randn(B,Hq,S,D,device='cuda',dtype=torch.bfloat16)
        K=torch.randn(B,Hkv,S,D,device='cuda',dtype=torch.bfloat16)
        V=torch.randn(B,Hkv,S,D,device='cuda',dtype=torch.bfloat16)
        sel=make_sel(B,Hkv,NQB,KSEL,BQ,LP,S)
        cyc=torch.zeros(NQB*B*Hkv*2,dtype=torch.int64,device='cuda')
        m1,l1,h1,_   = timed(lambda: v1.forward(Q,K,V,sel,LP,BQ,cyc))
        m2,l2,h2,ck  = timed(lambda: v2.forward(Q,K,V,sel,LP,BQ,cyc))
        w=cyc[0::2].float().mean().item(); mm=cyc[1::2].float().mean().item()
        waitpct=100*w/(w+mm)

        # effective FLOPs: only selected, causally-valid blocks are computed
        nsel_avg=(sel>=0).float().sum().item()/(B*Hkv*NQB)
        flops = 2*2*B*Hq*NQB*BQ*nsel_avg*LP*D
        # KV bytes are loaded once per CTA (Q tile is resident), so the
        # denominator counts blocks per CTA, not per query row.
        bytes_ = 2*2*B*Hkv*NQB*nsel_avg*LP*D          # K+V, bf16
        # bytes per selected block per CTA; FLOPs per those bytes
        ai = (2*2*BQ*G*LP*D) / (2*2*LP*D)             # == BQ*G, degenerate here
        ai_eff = flops/bytes_
        tf = flops/(l2*1e-3)/1e12
        rows.append(dict(G=G,BQ=BQ,D=D,LP=LP,S=S,KSEL=KSEL,Hq=Hq,
            v1_med=round(m1,4),v1_min=round(l1,4),
            v2_med=round(m2,4),v2_min=round(l2,4),
            speedup=round(m1/m2,3), wait_pct=round(waitpct,2),
            nsel_avg=round(nsel_avg,2), arith_intensity=round(ai,2), bytes_per_stage=2*2*LP*D,
            tflops=round(tf,1), gbs=round(bytes_/(l2*1e-3)/1e9,1),
            spread_pct=round(100*(h2-l2)/m2,1), sm_clk=ck))
        print(f"{G:>3}{BQ:>4}{D:>5}{LP:>5}{S:>7}{KSEL:>4}"
              f"{m1:9.3f}{m2:9.3f}{m1/m2:6.2f}x{waitpct:7.1f}{ai:8.1f}{tf:8.1f}")
        del Q,K,V,sel,cyc
    except RuntimeError as e:
        print(f"{G:>3}{BQ:>4}{D:>5}{LP:>5}{S:>7}{KSEL:>4}   SKIP {type(e).__name__}")
    torch.cuda.empty_cache()

out='/workspace/nsa/results/sweep.csv'
os.makedirs(os.path.dirname(out),exist_ok=True)
with open(out,'w',newline='') as f:
    w_=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w_.writeheader(); w_.writerows(rows)
print(f"\n{len(rows)} rows -> {out}   ({(time.time()-t_start)/60:.1f} min)")
