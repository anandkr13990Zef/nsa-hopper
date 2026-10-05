#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <cutlass/numeric_types.h>
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
using namespace cute;
using bf16 = cutlass::bfloat16_t;      // MUST match ValTypeA/B, not __nv_bfloat16

template<int D, int LP>
__global__ __launch_bounds__(128) void nsa_sel_v1(
    const bf16* __restrict__ Qp, const bf16* __restrict__ Kp,
    const bf16* __restrict__ Vp, const int* __restrict__ sel,
    float* __restrict__ O, int64_t* __restrict__ cyc,
    int B,int Hq,int Hkv,int S,int NQB,int KSEL,int BQ,float scale)
{
  extern __shared__ __align__(1024) char smem[];
  auto lQ = tile_to_shape(GMMA::Layout_K_SW128_Atom<bf16>{},  Shape<Int<64>,Int<D>>{});
  auto lK = tile_to_shape(GMMA::Layout_K_SW128_Atom<bf16>{},  Shape<Int<LP>,Int<D>>{});
  auto lV = tile_to_shape(GMMA::Layout_MN_SW128_Atom<bf16>{}, Shape<Int<D>,Int<LP>>{});
  auto lP = tile_to_shape(GMMA::Layout_K_SW128_Atom<bf16>{},  Shape<Int<64>,Int<LP>>{});

  bf16* p = (bf16*)smem;
  auto tQ = make_tensor(make_smem_ptr(p), lQ); p += cosize(lQ);
  auto tK = make_tensor(make_smem_ptr(p), lK); p += cosize(lK);
  auto tV = make_tensor(make_smem_ptr(p), lV); p += cosize(lV);
  auto tP = make_tensor(make_smem_ptr(p), lP);

  const int qb=blockIdx.x, bkv=blockIdx.y;
  const int b=bkv/Hkv, kvh=bkv%Hkv, G=Hq/Hkv, tid=threadIdx.x;

  for (int i=tid; i<64*D; i+=128) {
    int r=i/D, c=i%D, qi=r/G, g=r%G, qpos=qb*BQ+qi;
    bool ok=(r<BQ*G)&&(qpos<S);
    tQ(r,c) = ok ? Qp[(((size_t)b*Hq + kvh*G+g)*S + qpos)*D + c] : bf16(0.f);
  }

  TiledMMA mmaS = make_tiled_mma(SM90_64x64x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K >{});
  TiledMMA mmaO = make_tiled_mma(SM90_64x64x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::MN>{});
  auto thrS = mmaS.get_thread_slice(tid);
  auto thrO = mmaO.get_thread_slice(tid);

  // two-step: partition, then make_fragment  (per cute tutorial wgmma_sm90.cu)
  auto rQ = thrS.make_fragment_A(thrS.partition_A(tQ));
  auto rK = thrS.make_fragment_B(thrS.partition_B(tK));
  auto rP = thrO.make_fragment_A(thrO.partition_A(tP));
  auto rV = thrO.make_fragment_B(thrO.partition_B(tV));
  auto accS = partition_fragment_C(mmaS, Shape<Int<64>,Int<LP>>{});
  auto accO = partition_fragment_C(mmaO, Shape<Int<64>,Int<D >>{});
  clear(accO);

  auto cS = thrS.partition_C(make_identity_tensor(Shape<Int<64>,Int<LP>>{}));
  auto cO = thrO.partition_C(make_identity_tensor(Shape<Int<64>,Int<D >>{}));
  const int r0S = get<0>(cS(0)), r0O = get<0>(cO(0));

  float m_[2]={-INFINITY,-INFINITY}, l_[2]={0.f,0.f};
  int64_t t_load=0,t_math=0,t0;
  const int* mysel = sel + ((size_t)b*Hkv+kvh)*NQB*KSEL + (size_t)qb*KSEL;
  __syncthreads();

  for (int s=0;s<KSEL;++s) {
    int blk=mysel[s]; if(blk<0) continue;
    int base=blk*LP;
    t0=clock64();
    for (int i=tid;i<LP*D;i+=128) {
      int r=i/D,c=i%D,pp=base+r; bool ok=(pp<S);
      size_t off=(((size_t)b*Hkv+kvh)*S+pp)*D+c;
      tK(r,c) = ok?Kp[off]:bf16(0.f);
      tV(c,r) = ok?Vp[off]:bf16(0.f);     // transposed: B operand is (N=D, K=LP)
    }
    __syncthreads();
    t_load += clock64()-t0;

    t0=clock64();
    clear(accS);
    warpgroup_arrive();
    cute::gemm(mmaS, rQ, rK, accS);
    warpgroup_commit_batch(); warpgroup_wait<0>();

    #pragma unroll
    for (int i=0;i<size(accS);++i) {
      int row=get<0>(cS(i)), col=get<1>(cS(i));
      int qpos=qb*BQ+row/G, kpos=base+col;
      accS(i) = (row<BQ*G && qpos<S && kpos<=qpos && kpos<S) ? accS(i)*scale : -INFINITY;
    }
    #pragma unroll
    for (int h=0;h<2;++h) {
      float mx=-INFINITY;
      for (int i=0;i<size(accS);++i)
        if (((get<0>(cS(i))==r0S)?0:1)==h) mx=fmaxf(mx,accS(i));
      for (int o=1;o<4;o<<=1) mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,o));
      float mn=fmaxf(m_[h],mx);
      float corr=(m_[h]==-INFINITY)?0.f:__expf(m_[h]-mn);
      float sum=0.f;
      for (int i=0;i<size(accS);++i) if(((get<0>(cS(i))==r0S)?0:1)==h){
        float pv=(accS(i)==-INFINITY)?0.f:__expf(accS(i)-mn); accS(i)=pv; sum+=pv; }
      for (int o=1;o<4;o<<=1) sum+=__shfl_xor_sync(0xffffffff,sum,o);
      l_[h]=l_[h]*corr+sum; m_[h]=mn;
      for (int i=0;i<size(accO);++i)
        if (((get<0>(cO(i))==r0O)?0:1)==h) accO(i)*=corr;
    }
    __syncthreads();
    #pragma unroll
    for (int i=0;i<size(accS);++i)
      tP(get<0>(cS(i)),get<1>(cS(i))) = bf16(accS(i));
    __syncthreads();

    warpgroup_arrive();
    cute::gemm(mmaO, rP, rV, accO);
    warpgroup_commit_batch(); warpgroup_wait<0>();
    __syncthreads();
    t_math += clock64()-t0;
  }

  #pragma unroll
  for (int i=0;i<size(accO);++i) {
    int row=get<0>(cO(i)), col=get<1>(cO(i));
    if (row>=BQ*G) continue;
    int qi=row/G,g=row%G,qpos=qb*BQ+qi; if(qpos>=S) continue;
    float l=l_[(row==r0O)?0:1]; float inv=(l>0.f)?1.f/l:0.f;
    O[(((size_t)b*Hq + kvh*G+g)*S + qpos)*D + col] = accO(i)*inv;
  }
  if(tid==0&&cyc){int c=blockIdx.y*gridDim.x+blockIdx.x; cyc[c*2]=t_load; cyc[c*2+1]=t_math;}
}

torch::Tensor nsa_sel_forward_v1(torch::Tensor Q,torch::Tensor K,torch::Tensor V,
    torch::Tensor sel,int64_t l_prime,int64_t BQ,torch::Tensor cyc){
  int B=Q.size(0),Hq=Q.size(1),S=Q.size(2),D=Q.size(3);
  int Hkv=K.size(1),NQB=sel.size(2),KSEL=sel.size(3);
  TORCH_CHECK(BQ*(Hq/Hkv)==64,"v1 needs BQ*(Hq/Hkv)==64, got ",BQ*(Hq/Hkv));
  auto O=torch::zeros({B,Hq,S,D},Q.options().dtype(torch::kFloat32));
  float scale=1.f/sqrtf((float)D); dim3 grid(NQB,B*Hkv);
#define L1(DD,LL){ size_t sm=(64*DD+2*LL*DD+64*LL)*sizeof(bf16); \
  cudaFuncSetAttribute(nsa_sel_v1<DD,LL>,cudaFuncAttributeMaxDynamicSharedMemorySize,sm); \
  nsa_sel_v1<DD,LL><<<grid,128,sm>>>((const bf16*)Q.data_ptr(),(const bf16*)K.data_ptr(), \
    (const bf16*)V.data_ptr(),sel.data_ptr<int>(),O.data_ptr<float>(), \
    cyc.numel()?cyc.data_ptr<int64_t>():nullptr,B,Hq,Hkv,S,NQB,KSEL,(int)BQ,scale); }
  if(D==64&&l_prime==64) L1(64,64)
  else if(D==128&&l_prime==64) L1(128,64)
  else if(D==64&&l_prime==128) L1(64,128)
  else if(D==128&&l_prime==128) L1(128,128)
  else if(D==64&&l_prime==256)  L1(64,256)
  else if(D==128&&l_prime==256) L1(128,256)
  else TORCH_CHECK(false,"unsupported (D,l_prime)");
  C10_CUDA_CHECK(cudaGetLastError());
  return O;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){ m.def("forward",&nsa_sel_forward_v1); }
