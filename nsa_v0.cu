#include <torch/extension.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAException.h>
#define NWARP 8
#define NTHREAD (NWARP*32)

// v0: selected-branch NSA forward. Scalar math, cooperative SMEM staging,
// group-centric (one CTA per (batch, kv_head, query_block); the KV tile is
// loaded once and reused by all G query heads). Correctness oracle for v1/v2.
template<int D, int LP>
__global__ void nsa_sel_v0(
    const __nv_bfloat16* __restrict__ Q,   // [B,Hq,S,D]
    const __nv_bfloat16* __restrict__ K,   // [B,Hkv,S,D]
    const __nv_bfloat16* __restrict__ V,
    const int* __restrict__ sel,           // [B,Hkv,NQB,KSEL]
    float* __restrict__ O,                 // [B,Hq,S,D]
    int64_t* __restrict__ cyc,           // [nCTA*2] {load_wait, math}
    int B,int Hq,int Hkv,int S,int NQB,int KSEL,int BQ,float scale)
{
    extern __shared__ char smem[];
    __nv_bfloat16* sK = (__nv_bfloat16*)smem;
    __nv_bfloat16* sV = sK + LP*D;
    float* acc = (float*)(sV + LP*D);
    int   R    = BQ * (Hq/Hkv);
    float* mrow = acc + R*D;
    float* lrow = mrow + R;

    const int qb  = blockIdx.x;
    const int bkv = blockIdx.y;
    const int b   = bkv / Hkv, kvh = bkv % Hkv;
    const int G   = Hq / Hkv;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int DV  = D / 32;                       // dims per lane

    for (int i = tid; i < R*D; i += NTHREAD) acc[i] = 0.f;
    for (int i = tid; i < R;   i += NTHREAD) { mrow[i] = -INFINITY; lrow[i] = 0.f; }
    __syncthreads();

    int64_t t_load = 0, t_math = 0, t0;
    const int* mysel = sel + ((size_t)b*Hkv + kvh)*NQB*KSEL + (size_t)qb*KSEL;

    for (int s = 0; s < KSEL; ++s) {
        int blk = mysel[s];
        if (blk < 0) continue;
        int base = blk * LP;

        t0 = clock64();
        for (int i = tid; i < LP*D; i += NTHREAD) {
            int r = i / D, c = i % D, p = base + r;
            bool ok = (p < S);
            size_t off = (((size_t)b*Hkv + kvh)*S + p)*D + c;
            sK[i] = ok ? K[off] : __float2bfloat16(0.f);
            sV[i] = ok ? V[off] : __float2bfloat16(0.f);
        }
        __syncthreads();
        t_load += clock64() - t0;

        t0 = clock64();
        for (int r = warp; r < R; r += NWARP) {
            int qi = r / G, g = r % G;
            int qpos = qb*BQ + qi;
            if (qpos >= S) continue;
            int qh = kvh*G + g;
            size_t qoff = (((size_t)b*Hq + qh)*S + qpos)*D;

            float qreg[DV];
            #pragma unroll
            for (int j = 0; j < DV; ++j) qreg[j] = __bfloat162float(Q[qoff + lane + j*32]);

            float m = mrow[r], l = lrow[r];
            for (int t = 0; t < LP; ++t) {
                int kpos = base + t;
                if (kpos > qpos || kpos >= S) continue;      // causal + bounds
                float dot = 0.f;
                #pragma unroll
                for (int j = 0; j < DV; ++j)
                    dot += qreg[j] * __bfloat162float(sK[t*D + lane + j*32]);
                #pragma unroll
                for (int off = 16; off; off >>= 1) dot += __shfl_xor_sync(0xffffffff, dot, off);
                float sc = dot * scale;

                float mn = fmaxf(m, sc);
                float corr = __expf(m - mn);
                float p = __expf(sc - mn);
                l = l * corr + p;
                #pragma unroll
                for (int j = 0; j < DV; ++j) {
                    int idx = r*D + lane + j*32;
                    acc[idx] = acc[idx]*corr + p*__bfloat162float(sV[t*D + lane + j*32]);
                }
                m = mn;
            }
            mrow[r] = m; lrow[r] = l;
        }
        __syncthreads();
        t_math += clock64() - t0;
    }

    for (int r = warp; r < R; r += NWARP) {
        int qi = r / G, g = r % G;
        int qpos = qb*BQ + qi;
        if (qpos >= S) continue;
        int qh = kvh*G + g;
        float l = lrow[r];
        float inv = (l > 0.f) ? 1.f/l : 0.f;          // fully-masked row -> 0
        size_t ooff = (((size_t)b*Hq + qh)*S + qpos)*D;
        #pragma unroll
        for (int j = 0; j < DV; ++j)
            O[ooff + lane + j*32] = acc[r*D + lane + j*32] * inv;
    }

    if (tid == 0 && cyc) {
        int c = blockIdx.y*gridDim.x + blockIdx.x;
        cyc[c*2] = t_load; cyc[c*2+1] = t_math;
    }
}

torch::Tensor nsa_sel_forward_v0(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                                 torch::Tensor sel, int64_t l_prime, int64_t BQ,
                                 torch::Tensor cyc) {
    int B=Q.size(0), Hq=Q.size(1), S=Q.size(2), D=Q.size(3);
    int Hkv=K.size(1), NQB=sel.size(2), KSEL=sel.size(3);
    auto O = torch::zeros({B,Hq,S,D}, Q.options().dtype(torch::kFloat32));
    int R = BQ * (Hq/Hkv);
    float scale = 1.f/sqrtf((float)D);
    dim3 grid(NQB, B*Hkv);

#define LAUNCH(DD,LL) { \
    size_t sm = 2*(size_t)LL*DD*sizeof(__nv_bfloat16) + (size_t)R*DD*sizeof(float) + 2*R*sizeof(float); \
    cudaFuncSetAttribute(nsa_sel_v0<DD,LL>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm); \
    nsa_sel_v0<DD,LL><<<grid,NTHREAD,sm>>>( \
        (const __nv_bfloat16*)Q.data_ptr(), (const __nv_bfloat16*)K.data_ptr(), \
        (const __nv_bfloat16*)V.data_ptr(), sel.data_ptr<int>(), O.data_ptr<float>(), \
        cyc.numel()? cyc.data_ptr<int64_t>():nullptr, \
        B,Hq,Hkv,S,NQB,KSEL,(int)BQ,scale); }

    if (D==64  && l_prime==64 ) LAUNCH(64,64)
    else if (D==128 && l_prime==64 ) LAUNCH(128,64)
    else if (D==64  && l_prime==128) LAUNCH(64,128)
    else if (D==128 && l_prime==128) LAUNCH(128,128)
    else TORCH_CHECK(false, "unsupported (D,l_prime)=(",D,",",l_prime,")");
    C10_CUDA_CHECK(cudaGetLastError());
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("forward", &nsa_sel_forward_v0); }
