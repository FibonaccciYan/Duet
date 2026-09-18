#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

template<class T> __device__ float cvt(T x);
template<> __device__ float cvt(__half x){return __half2float(x);}
template<> __device__ float cvt(__nv_bfloat16 x){return __bfloat162float(x);}
template<class T> __device__ T cast(float x);
template<> __device__ __half cast(float x){return __float2half_rn(x);}
template<> __device__ __nv_bfloat16 cast(float x){return __float2bfloat16_rn(x);}

template<class T,int SHARE>
__global__ void shared_kernel(const T*Q,const T*K,const T*V,const int64_t*P,T*O,float*L,
 int A,int H,int HK,int N,int CAP,float scale,
 int qr,int qh,int qd,int kt,int kh,int kd,int vt,int vh,int vd,int ph,int ps){
 __shared__ float keys[16*128],values[16*128],probs[SHARE][16],scores[SHARE][16];
 __shared__ int count;
 const int head=blockIdx.x, warp=threadIdx.x/32,lane=threadIdx.x%32,l16=lane%16;
 const int g=H/HK, m=blockIdx.y*SHARE+warp,row=m/g,qhead=head*g+m%g;
 float q[8];
 #pragma unroll
 for(int j=0;j<8;j++)q[j]=row<A?cvt(Q[row*qr+qhead*qh+(l16*8+j)*qd]):0.f;
 if(threadIdx.x==0){int c=0;for(int s=0;s<CAP;s++)c+=P[head*ph+s*ps]>=0;count=c;}
 __syncthreads();
 float maximum=-INFINITY,normalizer=0.f,acc[4]={0.f,0.f,0.f,0.f};
 for(int slot=0;slot<count;slot++){
   int page=P[head*ph+slot*ps];
   for(int i=threadIdx.x;i<2048;i+=32*SHARE){
     int token=page*16+i/128,d=i%128;
     bool valid=page>=0&&token<N;
     keys[((i/128)*128+(d%8)*16+d/8)^(((i/128)&1)*16)]=valid?cvt(K[token*kt+head*kh+d*kd]):0.f;
     values[(i/128)*128+(d%4)*32+d/4]=valid?cvt(V[token*vt+head*vh+d*vd]):0.f;
   }
   __syncthreads();
   #pragma unroll
   for(int pair=0;pair<8;pair++){
     int n=pair*2+lane/16;
     float key[8];
     #pragma unroll
     for(int j=0;j<8;j++)key[j]=keys[(n*128+j*16+l16)^((n&1)*16)];
     float dot=__fmaf_rn(q[0],key[0],__fmul_rn(q[1],key[1]));
     #pragma unroll
     for(int j=2;j<8;j++)dot=__fmaf_rn(q[j],key[j],dot);
     #pragma unroll
     for(int off=8;off>0;off/=2)dot=__fadd_rn(dot,__shfl_xor_sync(0xffffffff,dot,off,16));
     if(l16==0)scores[warp][n]=(page*16+n<N)?__fmul_rn(dot,scale):-INFINITY;
   }
   __syncwarp();
   float s=scores[warp][l16],tilemax=s;
   #pragma unroll
   for(int off=8;off>0;off/=2)tilemax=fmaxf(tilemax,__shfl_xor_sync(0xffffffff,tilemax,off,16));
   float next=fmaxf(maximum,tilemax),alpha=__expf(maximum-next);
   float p=(page*16+l16<N)?__expf(s-next):0.f;
   if(lane<16)probs[warp][lane]=p;
   float sum=__fadd_rn(p,__shfl_xor_sync(0xffffffff,p,8,16));
   sum=__fadd_rn(sum,__shfl_xor_sync(0xffffffff,sum,1,16));
   sum=__fadd_rn(sum,__shfl_xor_sync(0xffffffff,sum,4,16));
   sum=__fadd_rn(sum,__shfl_xor_sync(0xffffffff,sum,2,16));
   sum=__shfl_sync(0xffffffff,sum,0);
   __syncwarp();
   #pragma unroll
   for(int j=0;j<4;j++){
     int d=j*32+lane;float y[8];
     #pragma unroll
     for(int t=0;t<8;t++)y[t]=__fmaf_rn(probs[warp][t],values[t*128+d],
                                      __fmul_rn(probs[warp][t+8],values[(t+8)*128+d]));
     float a=__fadd_rn(y[0],y[1]),b=__fadd_rn(y[2],y[3]);
     float c=__fadd_rn(y[4],y[5]),e=__fadd_rn(y[6],y[7]);
     float value=__fadd_rn(__fadd_rn(a,c),__fadd_rn(b,e));
     acc[j]=__fmaf_rn(acc[j],alpha,value);
   }
   normalizer=__fmaf_rn(normalizer,alpha,sum);maximum=next;
   __syncthreads();
 }
 if(row<A){
   #pragma unroll
   for(int j=0;j<4;j++)O[(row*H+qhead)*128+lane*4+j]=cast<T>(acc[j]/normalizer);
   if(lane==0)L[row*H+qhead]=maximum+logf(normalizer);
 }
}

template<class T> void dispatch(torch::Tensor q,torch::Tensor k,torch::Tensor v,
 torch::Tensor p,torch::Tensor o,torch::Tensor l,int n,float scale,int share){
 int a=q.size(0),h=q.size(1),hk=k.size(1),g=h/hk;
 dim3 grid(hk,(a*g+share-1)/share);
 auto stream=at::cuda::getCurrentCUDAStream();
 #define LAUNCH(S) shared_kernel<T,S><<<grid,32*S,0,stream>>>( \
 (T*)q.data_ptr(),(T*)k.data_ptr(),(T*)v.data_ptr(),p.data_ptr<int64_t>(), \
 (T*)o.data_ptr(),l.data_ptr<float>(),a,h,hk,n,p.size(1),scale, \
 q.stride(0),q.stride(1),q.stride(2),k.stride(0),k.stride(1),k.stride(2), \
 v.stride(0),v.stride(1),v.stride(2),p.stride(0),p.stride(1))
 if(share==1){LAUNCH(1);}else if(share==2){LAUNCH(2);}else{LAUNCH(4);}
 #undef LAUNCH
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void run(torch::Tensor q,torch::Tensor k,torch::Tensor v,torch::Tensor p,
 torch::Tensor o,torch::Tensor l,int64_t n,double scale,int64_t share){
 TORCH_CHECK(q.is_cuda()&&k.is_cuda()&&v.is_cuda()&&p.is_cuda(),"CUDA required");
 TORCH_CHECK(q.device()==k.device()&&q.device()==v.device()&&q.device()==p.device(),"device mismatch");
 TORCH_CHECK(q.dim()==3&&k.dim()==3&&v.sizes()==k.sizes()&&q.size(2)==128&&k.size(2)==128,"D128 layout");
 TORCH_CHECK(p.scalar_type()==at::kLong&&p.dim()==2&&p.size(0)==k.size(1),"page layout");
 TORCH_CHECK(q.size(1)%k.size(1)==0&&k.scalar_type()==q.scalar_type()&&v.scalar_type()==q.scalar_type(),"GQA dtype");
 TORCH_CHECK(share==1||share==2||share==4,"share");
 c10::cuda::CUDAGuard guard(q.device());
 if(q.scalar_type()==at::kHalf)dispatch<__half>(q,k,v,p,o,l,n,scale,share);
 else if(q.scalar_type()==at::kBFloat16)dispatch<__nv_bfloat16>(q,k,v,p,o,l,n,scale,share);
 else TORCH_CHECK(false,"half/bfloat16 required");
}
