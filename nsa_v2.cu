#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <cutlass/numeric_types.h>
#ifndef STOVR
#define STOVR 2
#endif
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
using namespace cute;
using bf16 = cutlass::bfloat16_t;

__device__ __forceinline__ void cp16(uint32_t dst, const void* src){
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(src));
}
__device__ __forceinline__ void cp_commit(){ asm volatile("cp.async.commit_group;\n"); }
template<int N> __device__ __forceinline__ void cp_wait(){
  asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}
__device__ __forceinline__ void cp_wait_all(){
  asm volatile("cp.async.wait_all;\n");
}

template<int D, int LP, int ST>
__global__ __launch_bounds__(128) void nsa_sel_v2(
    const bf16* __restrict__ Qp, const bf16* __restrict__ Kp,
    const bf16* __restrict__ Vp, const int* __restrict__ sel,
    float* __restrict__ O, int64_t* __restrict__ cyc,
    int B,int Hq,int Hkv,int S,int NQB,int KSEL,int BQ,float scale)
{
  extern __shared__ __align__(1024) char smem[];
  auto lQ = tile_to_shape(GMMA::Layout_K_SW128_Atom<bf16>{},  Shape<Int<64>,Int<D>>{});
  auto lP = tile_to_shape(GMMA::Layout_K_SW128_Atom<bf16>{},  Shape<Int<64>,Int<LP>>{});
  auto lK = tile_to_shape(GMMA::Layout_K_SW128_Atom<bf16>{},  Shape<Int<LP>,Int<D>,Int<ST>>{});
  auto lV = tile_to_shape(GMMA::Layout_MN_SW128_Atom<bf16>{}, Shape<Int<D>,Int<LP>,Int<ST>>{});

  bf16* p=(bf16*)smem;
  auto sQ=make_tensor(make_smem_ptr(p),lQ); p+=cosize(lQ);
  auto sP=make_tensor(make_smem_ptr(p),lP); p+=cosize(lP);
  auto sK=make_tensor(make_smem_ptr(p),lK); p+=cosize(lK);
  auto sV=make_tensor(make_smem_ptr(p),lV);

  const int qb=blockIdx.x, bkv=blockIdx.y;
  const int b=bkv/Hkv, kvh=bkv%Hkv, G=Hq/Hkv, tid=threadIdx.x;
  const size_t kvbase=((size_t)b*Hkv+kvh)*S*D;

  for (int i=tid;i<64*D;i+=128){
    int r=i/D,c=i%D,qi=r/G,g=r%G,qpos=qb*BQ+qi;
    bool ok=(r<BQ*G)&&(qpos<S);
    sQ(r,c)= ok?Qp[(((size_t)b*Hq+kvh*G+g)*S+qpos)*D+c]:bf16(0.f);
  }

  // stage a KV block: 16B (8 elem) vector copies; swizzle preserves 16B chunks
  auto stage=[&](int blk,int pipe){
    int base=blk*LP;
    for (int i=tid*8;i<LP*D;i+=128*8){
      int r=i/D, c=i%D;
      cp16(cast_smem_ptr_to_uint(&sK(r,c,pipe)), Kp+kvbase+(size_t)(base+r)*D+c);
      cp16(cast_smem_ptr_to_uint(&sV(c,r,pipe)), Vp+kvbase+(size_t)(base+r)*D+c);
    }
    cp_commit();
  };

  TiledMMA mmaS=make_tiled_mma(SM90_64x64x16_F32BF16BF16_SS<GMMA::Major::K,GMMA::Major::K >{});
  TiledMMA mmaO=make_tiled_mma(SM90_64x64x16_F32BF16BF16_SS<GMMA::Major::K,GMMA::Major::MN>{});
  auto thrS=mmaS.get_thread_slice(tid);
  auto thrO=mmaO.get_thread_slice(tid);
  auto rQ =thrS.make_fragment_A(thrS.partition_A(sQ));
  auto rP =thrO.make_fragment_A(thrO.partition_A(sP));
  auto rKs=thrS.make_fragment_B(thrS.partition_B(sK));
  auto rVs=thrO.make_fragment_B(thrO.partition_B(sV));
  auto accS=partition_fragment_C(mmaS,Shape<Int<64>,Int<LP>>{});
  auto accO=partition_fragment_C(mmaO,Shape<Int<64>,Int<D >>{});
  clear(accO);
  auto cS=thrS.partition_C(make_identity_tensor(Shape<Int<64>,Int<LP>>{}));
  auto cO=thrO.partition_C(make_identity_tensor(Shape<Int<64>,Int<D >>{}));
  const int r0S=get<0>(cS(0)), r0O=get<0>(cO(0));

  const int* mysel=sel+((size_t)b*Hkv+kvh)*NQB*KSEL+(size_t)qb*KSEL;
  int nsel=0; for(int s=0;s<KSEL;++s) if(mysel[s]>=0) nsel++;

  int issued=0;
  #pragma unroll
  for (int s=0;s<ST;++s) if (s<nsel){ stage(mysel[s],s); issued++; }

  float m_[2]={-INFINITY,-INFINITY}, l_[2]={0.f,0.f};
  int64_t t_wait=0,t_math=0,t0;

  for (int s=0;s<nsel;++s){
    int pipe=s%ST, base=mysel[s]*LP;
    t0=clock64();
    // wait_group<N> only blocks while >N groups are outstanding; near the
    // pipeline ends fewer than ST are in flight, so fall back to wait_all.
    if (issued - s >= ST) cp_wait<ST-1>(); else cp_wait_all();
    __syncthreads();
    t_wait+=clock64()-t0;

    t0=clock64();
    clear(accS);
    warpgroup_arrive();
    cute::gemm(mmaS,rQ,rKs(_,_,_,pipe),accS);
    warpgroup_commit_batch(); warpgroup_wait<0>();

    #pragma unroll
    for(int i=0;i<size(accS);++i){
      int row=get<0>(cS(i)),col=get<1>(cS(i));
      int qpos=qb*BQ+row/G,kpos=base+col;
      accS(i)=(row<BQ*G&&qpos<S&&kpos<=qpos&&kpos<S)?accS(i)*scale:-INFINITY;
    }
    #pragma unroll
    for(int h=0;h<2;++h){
      float mx=-INFINITY;
      for(int i=0;i<size(accS);++i) if(((get<0>(cS(i))==r0S)?0:1)==h) mx=fmaxf(mx,accS(i));
      for(int o=1;o<4;o<<=1) mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,o));
      float mn=fmaxf(m_[h],mx),corr=(m_[h]==-INFINITY)?0.f:__expf(m_[h]-mn),sum=0.f;
      for(int i=0;i<size(accS);++i) if(((get<0>(cS(i))==r0S)?0:1)==h){
        float pv=(accS(i)==-INFINITY)?0.f:__expf(accS(i)-mn); accS(i)=pv; sum+=pv; }
      for(int o=1;o<4;o<<=1) sum+=__shfl_xor_sync(0xffffffff,sum,o);
      l_[h]=l_[h]*corr+sum; m_[h]=mn;
      for(int i=0;i<size(accO);++i) if(((get<0>(cO(i))==r0O)?0:1)==h) accO(i)*=corr;
    }
    __syncthreads();
    #pragma unroll
    for(int i=0;i<size(accS);++i) sP(get<0>(cS(i)),get<1>(cS(i)))=bf16(accS(i));
    __syncthreads();

    warpgroup_arrive();
    cute::gemm(mmaO,rP,rVs(_,_,_,pipe),accO);
    warpgroup_commit_batch(); warpgroup_wait<0>();
    __syncthreads();
    t_math+=clock64()-t0;

    int nxt=s+ST;
    if (nxt<nsel){ stage(mysel[nxt],pipe); issued++; }
  }

  #pragma unroll
  for(int i=0;i<size(accO);++i){
    int row=get<0>(cO(i)),col=get<1>(cO(i));
    if(row>=BQ*G) continue;
    int qi=row/G,g=row%G,qpos=qb*BQ+qi; if(qpos>=S) continue;
    float l=l_[(row==r0O)?0:1],inv=(l>0.f)?1.f/l:0.f;
    O[(((size_t)b*Hq+kvh*G+g)*S+qpos)*D+col]=accO(i)*inv;
  }
  if(tid==0&&cyc){int c=blockIdx.y*gridDim.x+blockIdx.x; cyc[c*2]=t_wait; cyc[c*2+1]=t_math;}
}

torch::Tensor nsa_sel_forward_v2(torch::Tensor Q,torch::Tensor K,torch::Tensor V,
    torch::Tensor sel,int64_t l_prime,int64_t BQ,torch::Tensor cyc){
  int B=Q.size(0),Hq=Q.size(1),S=Q.size(2),D=Q.size(3);
  int Hkv=K.size(1),NQB=sel.size(2),KSEL=sel.size(3);
  TORCH_CHECK(BQ*(Hq/Hkv)==64,"v2 needs BQ*(Hq/Hkv)==64");
  TORCH_CHECK(S%l_prime==0,"v2 needs S divisible by l_prime");
  auto O=torch::zeros({B,Hq,S,D},Q.options().dtype(torch::kFloat32));
  float scale=1.f/sqrtf((float)D); dim3 grid(NQB,B*Hkv);
#define L2(DD,LL,SS){ size_t sm=(64*DD+64*LL+2*SS*LL*DD)*sizeof(bf16); \
  cudaFuncSetAttribute(nsa_sel_v2<DD,LL,SS>,cudaFuncAttributeMaxDynamicSharedMemorySize,sm); \
  nsa_sel_v2<DD,LL,SS><<<grid,128,sm>>>((const bf16*)Q.data_ptr(),(const bf16*)K.data_ptr(), \
    (const bf16*)V.data_ptr(),sel.data_ptr<int>(),O.data_ptr<float>(), \
    cyc.numel()?cyc.data_ptr<int64_t>():nullptr,B,Hq,Hkv,S,NQB,KSEL,(int)BQ,scale); }
  if(D==64&&l_prime==64)        L2(64,64,1)
  else if(D==128&&l_prime==64)  L2(128,64,1)
  else if(D==64&&l_prime==128)  L2(64,128,1)
  else if(D==128&&l_prime==128) L2(128,128,STOVR)
  else if(D==64&&l_prime==256)  L2(64,256,1)
  else if(D==128&&l_prime==256) L2(128,256,1)
  else TORCH_CHECK(false,"unsupported (D,l_prime)");
  C10_CUDA_CHECK(cudaGetLastError());
  return O;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){ m.def("forward",&nsa_sel_forward_v2); }
