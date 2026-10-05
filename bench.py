"""CUDA-graph-captured timing. Median-of-N, warmup discarded, clock logged."""
import torch, time, csv, subprocess, statistics, os
from torch.nn.attention import SDPBackend, sdpa_kernel

def sm_clock():
    try:
        return int(subprocess.check_output(
            "nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits",
            shell=True).decode().strip().split('\n')[0])
    except Exception: return -1

def timeit(fn, warmup=20, iters=100, graph=True, cooldown=0.5):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    if graph:
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3): fn()
        torch.cuda.current_stream().wait_stream(s)
        with torch.cuda.graph(g): fn()
        run = g.replay
    else:
        run = fn
    import time as _t; _t.sleep(cooldown)
    ts, clks = [], []
    for _ in range(iters):
        st = torch.cuda.Event(True); en = torch.cuda.Event(True)
        st.record(); run(); en.record(); torch.cuda.synchronize()
        ts.append(st.elapsed_time(en)); clks.append(sm_clock())
    return statistics.median(ts), min(ts), max(ts), min(clks), max(clks)

def sdpa_baseline(B,Hq,Hkv,S,D,backend):
    G = Hq//Hkv
    q = torch.randn(B,Hq,S,D, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(B,Hkv,S,D,device='cuda', dtype=torch.bfloat16).repeat_interleave(G,1)
    v = torch.randn(B,Hkv,S,D,device='cuda', dtype=torch.bfloat16).repeat_interleave(G,1)
    def f():
        with sdpa_kernel(backend):
            torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=True)
    return f

def dense_flops(B,Hq,S,D):   # causal: half the matrix, two matmuls
    return 2 * 2 * B * Hq * (S*S/2) * D

if __name__ == '__main__':
    out = '/workspace/nsa/results/baselines.csv'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rows = []
    for S in [4096, 8192, 16384, 32768]:
        for D in [64, 128]:
            B,Hq,Hkv = 1, 32, 4
            for name, bk in [('cudnn',SDPBackend.CUDNN_ATTENTION),
                             ('flash',SDPBackend.FLASH_ATTENTION)]:
                try:
                    med,lo,hi,ck_lo,ck_hi = timeit(sdpa_baseline(B,Hq,Hkv,S,D,bk))
                    tf = dense_flops(B,Hq,S,D)/(med*1e-3)/1e12
                    rows.append(dict(impl=name,B=B,Hq=Hq,Hkv=Hkv,S=S,D=D,
                                     ms_med=round(med,4), ms_min=round(lo,4),
                                     ms_max=round(hi,4), tflops=round(tf,1),
                                     clk_lo=ck_lo, clk_hi=ck_hi))
                    print(f"{name:6s} S={S:6d} D={D:3d}  {med:7.3f} ms  {tf:6.1f} TF/s  "
                          f"clk {ck_lo}-{ck_hi}MHz  spread {(hi-lo)/med*100:.1f}%")
                except Exception as e:
                    print(f"{name:6s} S={S:6d} D={D:3d}  FAIL {type(e).__name__}")
        torch.cuda.empty_cache()
    with open(out,'w',newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out}")
