# nsa-hopper

Hand-written CUDA kernels for the selected branch of [NSA](https://arxiv.org/abs/2502.11089)-style trainable sparse attention on H200 (SM90a), plus a 284-config sweep and a controlled ablation on shared-memory staging.

**Headline:** for scattered-block sparse attention, spend shared memory on larger blocks, not deeper pipelines. At fixed block size, going from 1 to 3 pipeline stages made the kernel **1.78× slower** — each stage's SMEM cost halves the number of resident CTAs per SM.

---

## The question

NSA argues that GQA-group-shared block selection keeps sparse attention compute-bound rather than gather-bound. The open engineering question is whether FlashAttention-3-style async pipelining survives when the KV blocks being loaded are non-contiguous in HBM.

I expected deeper pipelines to help. They don't.

## Kernels

All three implement the *selected* branch only: one CTA per (batch, kv-head, query-block), with the KV tile loaded once and reused across all `G` query heads in the group. The compressed and sliding-window branches stay in PyTorch — they are dense and local, and don't touch the scatter question.

| | Compute | KV load | err vs fp64 |
|---|---|---|---|
| `src/nsa_v0.cu` | scalar CUDA cores | elementwise, sync | 2.4e-07 (fp32 acc) |
| `src/nsa_v1.cu` | `wgmma` via CuTe | elementwise, sync | 1.0e-03 (bf16) |
| `src/nsa_v2.cu` | `wgmma` via CuTe | 16B `cp.async.cg`, staged | 1.0e-03 (bf16) |

v0 exists to be obviously correct, not fast. It is the numerical oracle for everything after it.

---

## Results

### 1. Pipeline depth is monotonically harmful

Controlled ablation — everything fixed except stage count (D=128, LP=128, S=16384, k=32):

| stages | time | SMEM/CTA | CTAs/SM |
|---|---|---|---|
| 1 | 7.04 ms | 96 KB | 2 |
| 2 | 10.52 ms | 160 KB | 1 |
| 3 | 12.53 ms | 224 KB | 1 |

The mechanism is occupancy, not prefetch. Staging both K and V costs `4·LP·D` bytes per stage; at D=128 the second stage pushes you from two resident CTAs to one. Inter-CTA concurrency was already hiding the load latency that intra-CTA prefetch was meant to hide, and it does so without spending shared memory.

At ST=4 the config is not launchable at all — 288 KB exceeds the 227 KB limit.

### 2. The v1→v2 speedup is coalescing, not pipelining

The gain comes from replacing v1's elementwise SMEM writes with 16-byte vector copies, not from the staging. Measured directly by the fraction of cycles spent in the load wait (in-kernel `clock64()` accumulators), which falls monotonically with transfer size:

| bytes/stage | wait fraction |
|---|---|
| 16 KB | ~12% |
| 32 KB | ~7% |
| 65 KB | ~1.7% |
| 131 KB | ~1.1% |

Peak v2/v1: **4.85×** (D=128, LP=256, S=32768, k=32).

### 3. `wgmma` M=64 flattens the GQA group axis

`wgmma.m64n64k16` fixes the accumulator's M dimension at 64, so the query-block size is forced to `BQ = 64/G`. Total KV bytes per CTA are therefore constant across group size — the group-sharing amortization that NSA's efficiency argument rests on cannot be varied within this design. Empirically, G=4 and G=16 land within noise at every matched config.

Breaking this needs either a different MMA tile or splitting the group across CTAs, both of which give up the single-load reuse that motivates the group-centric layout in the first place.

---

## Validation

Every kernel is checked against a 200-case frozen test set with fp64 ground truth, covering fully-masked rows, selection sets smaller than `n`, sequence lengths not divisible by block size, group sizes 1 and 16, and large-magnitude query rows.

Metric is `max|err| / max|out|` (scale-relative). Element-relative error is meaningless here because attention outputs pass near zero.

Precision floors, measured from the reference itself at reduced precision — no kernel can beat these:

| | floor |
|---|---|
| fp32 accumulate | 2e-06 |
| bf16 | 3e-02 |

---

## Limitations

Stated plainly, because they bound what the results claim.

- **Forward only.** No backward pass, so no training run and no quality claim. These kernels measure; they don't train.
- **No Nsight Compute.** The rented instance had `RmProfilingAdminOnly=1` at the driver level, so no stall-reason histogram. Mechanism data comes from in-kernel `clock64()` accumulators instead — coarser than `ncu`, but it measures the specific quantity the claim needs.
- **No clock lock.** `nvidia-smi -lgc` needs `CAP_SYS_ADMIN`, unavailable in the container. Mitigated with median-of-60, 20 discarded warmups, and per-config SM-clock logging. Clock held at 1980 MHz with sub-3% spread on most configs; `spread_pct` is in the CSV.
- **No TMA.** The CuTe TMA descriptor path did not compile against the CUTLASS version on hand, and `cp.async` answered the same question. TMA's SMEM cost per stage is lower than `cp.async` double-buffering of K and V, so finding #1 may not transfer to a TMA implementation — that is the obvious next experiment.
- **Not compared to production kernels.** `results/baselines.csv` has dense cuDNN and FlashAttention SDPA numbers from the same machine for context, but sparse and dense compute different FLOP counts and no iso-work comparison was made. Nothing here claims parity with cuDNN.

---

## Files

```
src/
  nsa_v0.cu  nsa_v1.cu  nsa_v2.cu     kernels
  nsa_reference.py                     full NSA forward, fp64
  ref_only.py                          selected-branch reference
  make_testset.py  validate.py         oracle + precision floors
  test_v0.py  test_v1.py  test_v2.py   correctness
  bench.py  cmp_v1v2.py                dense baselines, v1-vs-v2
  sweep.py  sweep2.py                  parameter sweeps
results/
  sweep2.csv                           284 rows, final sweep at ST=1
  sweep.csv                            first pass, superseded
  baselines.csv                        dense cuDNN / FA SDPA
logs/machine_state.txt                 driver, CUDA, clocks at measurement time
```

`sweep.csv` is kept for completeness but was run at the old staging depths; its `arith_intensity` column is constant by construction (it reduces to `BQ·G`, which is pinned at 64). Use `sweep2.csv`.

## Reproducing

```bash
python src/make_testset.py     # fp64 oracle, 200 cases
python src/validate.py         # precision floors
python src/test_v2.py          # kernel correctness
python src/sweep2.py           # 284-config sweep, ~5 min
```

H200, CUDA 12.6, PyTorch 2.13+cu126, CUTLASS 3.x.

`-arch=sm_90a` is required. Plain `sm_90` compiles cleanly and silently omits `wgmma`, which produces a kernel that runs correctly and is slow for no visible reason.
