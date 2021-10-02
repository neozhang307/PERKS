#ifdef _TIMER_
#include "cuda_profiler_api.h"
#endif
#include <cuda.h>
#include "stdio.h"
#include <cooperative_groups.h>
#include "stdio.h"
#include "assert.h"
#include "config.cuh" 
#include "./common/jacobi_cuda.cuh"
#include "./common/types.hpp"
#include "./common/cuda_header.cuh"

#ifdef SMASYNC
  #if PERKS_ARCH<800 
    #error "unsupport architecture"
  #endif
  #include <cooperative_groups/memcpy_async.h>
  #include <cuda_pipeline.h>
#endif


//#ifndef REAL
//#define REAL float
//#endif
//configuration
#if defined(NAIVE)||defined(BASELINE)||defined(BASELINE_CM)
  #define TRADITIONLAUNCH
#endif
#if defined(GEN)||defined(MIX)||defined(PERSISTENT)
  #define PERSISTENTLAUNCH
#endif
#if defined PERSISTENTLAUNCH||defined(BASELINE_CM)
  #define PERSISTENTTHREAD
#endif
#if defined(BASELINE)||defined(BASELINE_CM)||defined(GEN)||defined(MIX)||defined(PERSISTENT)
  #define USEMAXSM
#endif

#ifdef __PRINT__ 
  #define WARMUPRUN
#endif


#define FORMA_MAX(a,b) ( (a) > (b) ? (a) : (b) )
#define MAX(a,b) FORMA_MAX(a,b)
#define FORMA_MIN(a,b) ( (a) < (b) ? (a) : (b) )
#define MIN(a,b) FORMA_MIN(a,b)
#define FORMA_CEIL(a,b) ( (a) % (b) == 0 ? (a) / (b) : ((a) / (b)) + 1 )


//#define SM_TILE_X (TILE_X+2*(HALO))
//#ifndef FORMA_MAX_BLOCKDIM_0
//#define FORMA_MAX_BLOCKDIM_0 1024
//#endif
//#ifndef FORMA_MAX_BLOCKDIM_1
//#define FORMA_MAX_BLOCKDIM_1 1024
//#endif
//#ifndef FORMA_MAX_BLOCKDIM_2
//#define FORMA_MAX_BLOCKDIM_2 1024
//#endif

namespace cg = cooperative_groups;

void Check_CUDA_Error(const char* message);

#ifndef RTILE_Y
#define RTILE_Y (8)
#endif

#ifndef SFOLDER_Y
#define SFOLDER_Y (2)
#endif
#ifndef RFOLDER_Y
#define RFOLDER_Y (9)
#endif

#define TSTILE_Y (RTILE_Y*SFOLDER_Y)
#define TRTILE_Y (RTILE_Y*RFOLDER_Y)
#define TILE_Y (TSTILE_Y+TRTILE_Y)

#define FOLDER (1)

#define bdim_x (TILE_X)

#define TILE_SM_X (TILE_X+Halo*2)
#define TILE_SM_Y (TSTILE_Y+Halo*2)


#define BOULDER_STEP (Halo*2*(TILE_Y+TILE_X))
#define L2_STEP (BOULDER_STEP)
#define L2_EW_STEP (Halo*2*(TILE_Y)) 

#define E_STEP (0)
#define W_STEP (TILE_Y*Halo)
#define S_STEP (TILE_Y*Halo*2)
#define N_STEP (TILE_Y*Halo*2+Halo*TILE_X)

#define cudaCheckError() {                                          \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
 }                                                                 \
}

#define stencilParaT \
  const REAL west[6]={12.0/118,9.0/118,3.0/118,2.0/118,5.0/118,6.0/118};\
  const REAL east[6]={12.0/118,9.0/118,3.0/118,3.0/118,4.0/118,6.0/118};\
  const REAL north[6]={5.0/118,7.0/118,5.0/118,4.0/118,3.0/118,2.0/118};\
  const REAL south[6]={5.0/118,7.0/118,5.0/118,1.0/118,6.0/118,2.0/118};\
  const REAL center=15.0/118;


#define COMPUTE(sm_ptrs,sm_idx,r_ptr,r_idx) \
do{\
  _Pragma("unroll")\
  for(int hl=0; hl<Halo; hl++)\
  {\
    _Pragma("unroll")\
    for(int l_y=0; l_y<RTILE_Y ; l_y++)\
    {\
      sum[l_y]+=sm_ptrs[(l_y+ps_y+sm_idx)][local_x+ps_x-1-hl]*west[hl];\
    }\
  }\
  _Pragma("unroll")\
  for(int hl=0; hl<Halo; hl++)\
  {\
    _Pragma("unroll")\
    for(int l_y=0; l_y<RTILE_Y ; l_y++)\
    {\
      sum[l_y]+=r_ptr[r_idx+l_y+Halo-1-hl]*south[hl];\
    }\
  }\
  _Pragma("unroll")\
  for(int l_y=0; l_y<RTILE_Y ; l_y++)\
  {\
    sum[l_y]+=r_ptr[r_idx+l_y+Halo]*center;\
  }\
  _Pragma("unroll")\
  for(int hl=0; hl<Halo; hl++)\
  {\
    _Pragma("unroll")\
    for(int l_y=0; l_y<RTILE_Y ; l_y++)\
    {\
      sum[l_y]+=r_ptr[r_idx+l_y+Halo+1+hl]*north[hl];\
    }\
  }\
  _Pragma("unroll")\
  for(int hl=0; hl<Halo; hl++)\
  {\
    _Pragma("unroll")\
    for(int l_y=0; l_y<RTILE_Y ; l_y++)\
    {\
      sum[l_y]+=sm_ptrs[(l_y+ps_y+sm_idx)][local_x+ps_x+1+hl]*east[hl];\
    }\
  }\
}while(0)

#define COMPUTE2(sm_ptr,sm_idx,r_ptr,r_idx) \
do{\
  _Pragma("unroll")\
  for(int hl=0; hl<Halo; hl++)\
  {\
    _Pragma("unroll")\
    for(int l_y=0; l_y<RTILE_Y ; l_y++)\
    {\
      sum[l_y]+=sm_ptr[(l_y+ps_y+sm_idx)*TILE_SM_X+local_x+ps_x-1-hl]*west[hl];\
    }\
  }\
  _Pragma("unroll")\
  for(int hl=0; hl<Halo; hl++)\
  {\
    _Pragma("unroll")\
    for(int l_y=0; l_y<RTILE_Y ; l_y++)\
    {\
      sum[l_y]+=r_ptr[r_idx+l_y+Halo-1-hl]*south[hl];\
    }\
  }\
  _Pragma("unroll")\
  for(int l_y=0; l_y<RTILE_Y ; l_y++)\
  {\
    sum[l_y]+=r_ptr[r_idx+l_y+Halo]*center;\
  }\
  _Pragma("unroll")\
  for(int hl=0; hl<Halo; hl++)\
  {\
    _Pragma("unroll")\
    for(int l_y=0; l_y<RTILE_Y ; l_y++)\
    {\
      sum[l_y]+=r_ptr[r_idx+l_y+Halo+1+hl]*north[hl];\
    }\
  }\
  _Pragma("unroll")\
  for(int hl=0; hl<Halo; hl++)\
  {\
    _Pragma("unroll")\
    for(int l_y=0; l_y<RTILE_Y ; l_y++)\
    {\
      sum[l_y]+=sm_ptr[(l_y+ps_y+sm_idx)*TILE_SM_X+local_x+ps_x+1+hl]*east[hl];\
    }\
  }\
}while(0)

template<class REAL, int RTILING, int halo>
__device__ void __forceinline__ computation(REAL result[RTILING], 
                                            REAL* sm_ptr, int sm_y_base, int sm_x_ind,int sm_width, 
                                            REAL r_ptr[RTILING+2*halo], int reg_base, 
                                            const REAL west[6],const REAL east[6], 
                                            const REAL north[6],const REAL south[6],
                                            const REAL center 
                                          )
{
  _Pragma("unroll")
  for(int hl=0; hl<halo; hl++)
  {
    _Pragma("unroll")
    for(int l_y=0; l_y<RTILING ; l_y++)
    {
      result[l_y]+=sm_ptr[(l_y+sm_y_base)*sm_width+sm_x_ind-1-hl]*west[hl];
    }
  }
  _Pragma("unroll")
  for(int hl=0; hl<halo; hl++)
  {
    _Pragma("unroll")
    for(int l_y=0; l_y<RTILING ; l_y++)
    {
      result[l_y]+=r_ptr[reg_base+l_y+halo-1-hl]*south[hl];
    }
  }
  _Pragma("unroll")
  for(int l_y=0; l_y<RTILING ; l_y++)
  {
    result[l_y]+=r_ptr[reg_base+l_y+halo]*center;
  }
  _Pragma("unroll")
  for(int hl=0; hl<halo; hl++)
  {
    _Pragma("unroll")
    for(int l_y=0; l_y<RTILING ; l_y++)
    {
      result[l_y]+=r_ptr[reg_base+l_y+halo+1+hl]*north[hl];
    }
  }
  _Pragma("unroll")
  for(int hl=0; hl<halo; hl++)
  {
    _Pragma("unroll")
    for(int l_y=0; l_y<RTILING ; l_y++)
    {
      result[l_y]+=sm_ptr[(l_y+sm_y_base)*sm_width+sm_x_ind+1+hl]*east[hl];
    }
  }
}


// init register array of ARRAY

template<class REAL, int SIZE>
__device__ void __forceinline__ init_reg_array(REAL reg_array[SIZE], int val)
{
  _Pragma("unroll")
  for(int l_y=0; l_y<SIZE; l_y++)
  {
    reg_array[l_y]=val;
  }
}

//load global memory src to shared memory
template<class REAL, int START, int END, bool isInit=false>
__device__ void __forceinline__ global2sm(REAL* src, REAL* sm_buffer, 
                                              int global_y, int global_y_end,
                                              int p_x, int width_x,
                                              int tid, int local_x,
                                              int ps_y, int ps_x, int sm_width)
{
  //fill shared memory buffer
  _Pragma("unroll")
  for(int l_y=START; l_y<END; l_y++)
  {
    int l_global_y;
    if(isInit)
    {
      l_global_y=(MAX(global_y+l_y,0));
    }
    else
    {
      l_global_y=(MIN(global_y+l_y,global_y_end-1));
      l_global_y=(MAX(l_global_y,0));
    }
    
  
    int dst_ind=(l_y+ps_y)*sm_width;
    
    #ifndef DA100X
      sm_buffer[dst_ind-Halo+local_x+ps_x]=src[l_global_y * width_x + MAX(p_x-Halo+local_x,0)];
      if(tid<Halo*2)
        sm_buffer[dst_ind-Halo+local_x+blockDim.x+ps_x]=src[(l_global_y) * width_x + MIN(-Halo+local_x+blockDim.x+p_x,width_x-1)];
    #else
      __pipeline_memcpy_async(sm_buffer+dst_ind-Halo+local_x+ps_x, 
            src + (l_global_y) * width_x + MAX(p_x-Halo+local_x,0)
              , sizeof(REAL));
      if(tid<Halo*2)
      {
        __pipeline_memcpy_async(sm_buffer+dst_ind-Halo+local_x+blockDim.x+ps_x, 
                src + (l_global_y) * width_x + MIN(-Halo+local_x+blockDim.x+p_x,width_x-1)
                  , sizeof(REAL));
      }
    #endif
  }
  #ifdef DA100X
    __pipeline_commit();
    __pipeline_wait_prior(0);
  #endif
}

//store register array to global memory dst
template<class REAL, int SIZE>
__device__ void __forceinline__ reg2global(REAL reg_array[SIZE], REAL* dst, 
  int global_y, int global_y_end, 
  int global_x, int width_x)
  {
    _Pragma("unroll")
    for(int l_y=0; l_y<SIZE; l_y++)
    {
      int l_global_y=global_y+l_y;
      if(l_global_y>=global_y_end)
      {
        break;
      }
      dst[(l_global_y) * width_x + global_x]=reg_array[l_y];
    }
}


template<class REAL, int START, int END>
__device__ void __forceinline__ smself(REAL *sm_buffer, 
                                      int ps_y, int y_step, 
                                      int local_x, int sm_width)
{
  _Pragma("unroll")
  for(int l_y=START; l_y<END; l_y++)
  {
    int dst_ind=(l_y+ps_y)*(sm_width);
    int src_ind=(l_y+ps_y+y_step)*(sm_width);
    sm_buffer[dst_ind+local_x]=sm_buffer[src_ind+local_x];
    if(threadIdx.x<Halo*2)
        sm_buffer[dst_ind+local_x+blockDim.x]=sm_buffer[src_ind+local_x+blockDim.x];

  }
}

template<class REAL, int START, int END, int SIZE>
__device__ void __forceinline__ sm2reg(REAL reg_array[SIZE], REAL* sm_buffer,
                                      int y_base, 
                                      int x_base, int x_id,
                                      int sm_width)
{
  _Pragma("unroll")
  for(int l_y=START; l_y<END ; l_y++)
  {
    reg_array[l_y] = sm_buffer[(l_y+y_base)*sm_width+x_base+x_id];//input[(global_y) * width_x + global_x];
  }
}

//__constant__ REAL center=15.0/118;
//__constant__ REAL west[6]={12.0/118,9.0/118,3.0/118,2.0/118,5.0/118,6.0/118};
//__constant__ REAL east[6]={12.0/118,9.0/118,3.0/118,3.0/118,4.0/118,6.0/118};
//__constant__ REAL north[6]={5.0/118,7.0/118,5.0/118,4.0/118,3.0/118,2.0/118};
//__constant__ REAL south[6]={5.0/118,7.0/118,5.0/118,1.0/118,6.0/118,2.0/118};


#if defined(BASELINE_CM)||defined(BASELINE)||defined(PERSISTENT)

template<class REAL>
#ifdef PERSISTENT
__global__ void kernel_persistent_baseline(REAL * __restrict__ input, int width_y, int width_x, 
  REAL * __restrict__ __var_4__,REAL * __restrict__ l2_cache=NULL, REAL * __restrict__ l2_cachetmp=NULL, 
  int iteration=0)
#else
__global__ void kernel_baseline(REAL * __restrict__ input, int width_y, int width_x, 
  REAL * __restrict__ __var_4__)
#endif

{

  stencilParaT;
  extern __shared__ char sm[];
  
  REAL* sm_space = (REAL*)sm+1;
  REAL* sm_rbuffer = sm_space;

  register REAL r_smbuffer[2*Halo+RTILE_Y];

  const int tid = threadIdx.x;
  // int ps_x = Halo + tid;
  const int ps_y = Halo;
  const int ps_x = Halo;
   // REAL* sb2 = sb+TILE_SM_X*TILE_SM_Y;
  const int p_x = blockIdx.x * TILE_X ;

  int blocksize_y=(width_y/gridDim.y);
  int y_quotient = width_y%gridDim.y;

  const int p_y =  blockIdx.y * (blocksize_y) + (blockIdx.y<=y_quotient?blockIdx.y:y_quotient);
  blocksize_y += (blockIdx.y<y_quotient?1:0);
  const int p_y_end = p_y + (blocksize_y);
 
  // for(int iter=0; iter<iteration; iter++)
#ifdef PERSISTENT  
  cg::grid_group gg = cg::this_grid();
  for(int iter=0; iter<iteration; iter++)
#endif
  {
    int local_x=tid;

    global2sm<REAL,-Halo,Halo,true>(input, sm_rbuffer, 
                                            p_y, width_y,
                                            p_x, width_x,
                                            tid, tid,
                                            ps_y, ps_x,TILE_SM_X);

    //computation of register space 
    for(int global_y=p_y; global_y<p_y_end; global_y+=RTILE_Y)
    {

      global2sm<REAL,Halo,RTILE_Y+Halo>(input, sm_rbuffer, 
                                            global_y, width_y,
                                            p_x, width_x,
                                            tid, tid,
                                            ps_y, ps_x,TILE_SM_X);

      __syncthreads();
      //shared memory buffer -> register buffer
      sm2reg<REAL,0,Halo*2+RTILE_Y, Halo*2+RTILE_Y>(r_smbuffer, sm_rbuffer,
                                                    0,
                                                    ps_x, tid,
                                                    TILE_SM_X);

      REAL sum[RTILE_Y];
      init_reg_array<REAL,RTILE_Y>(sum,0);
      //main computation
      //COMPUTE2(sm_rbuffer,0,r_smbuffer,0);
      
      computation<REAL,RTILE_Y,Halo>(sum,
                                      sm_rbuffer, ps_y, local_x+ps_x, TILE_SM_X,
                                      r_smbuffer, 0,
                                      west, east,
                                      north,south,  center);

      //store to global
      reg2global<REAL,RTILE_Y>(sum, __var_4__, 
                  global_y,p_y_end, 
                  p_x+local_x, width_x);
      __syncthreads();
      
      //some data in shared memroy can be used in next tiling. 
      smself<REAL,-Halo, Halo>(sm_rbuffer, ps_y, RTILE_Y, tid, TILE_SM_X);

    }

    #ifdef PERSISTENT
      if(iter==iteration-1)break;
  
      gg.sync();

      REAL* tmp_ptr =__var_4__;
      __var_4__=input;
      input=tmp_ptr;
    #endif
  }
} 
#endif

#ifdef NAIVE
template<class REAL>
__global__ void kernel2d_restrict(REAL* input,
                                  int width_y, int width_x, REAL* output) 
{
  stencilParaT;
  int l_x = blockDim.x * blockIdx.x + threadIdx.x;  
  int l_y = blockDim.y * blockIdx.y + threadIdx.y;
  int c = l_x + l_y * width_x;
  int w[Halo];
  int e[Halo];
  int n[Halo];
  int s[Halo];
   
  _Pragma("unroll") 
  for(int hl=0; hl<Halo; hl++)
  {
    w[hl] = MAX(0,l_x-1-hl)+l_y * width_x;
    e[hl] = MIN(width_x-1,l_x+1+hl)+l_y * width_x;
    s[hl] = l_x+MAX(0,l_y-1-hl) * width_x;;
    n[hl] = l_x+MIN(width_y-1,l_y+1+hl) * width_x;
  }
  REAL sum=0;
   
  _Pragma("unroll") 
  for(int hl=0; hl<Halo; hl++)
  {
    sum+=south[hl]*input[s[hl]];
  }
   
  _Pragma("unroll") 
  for(int hl=0; hl<Halo; hl++)
  {
    sum+=west[hl]*input[w[hl]];
  }
  sum+=center*input[c];
   
  _Pragma("unroll") 
  for(int hl=0; hl<Halo; hl++)
  {
    sum+=east[hl]*input[e[hl]];
  }
   
  _Pragma("unroll") 
  for(int hl=0; hl<Halo; hl++)
  {
    sum+=north[hl]*input[n[hl]];
  }
  output[c]=sum;

  return;
}
#endif

  __global__ void printptx()
  {
    printf("code is run in %d\n",PERKS_ARCH);
  }
  void host_printptx()
  {
    printptx<<<1,1>>>();
    cudaDeviceSynchronize();
  }

template<class REAL>
void jacobi_iterative(REAL * h_input, int width_y, int width_x, REAL * __var_0__, int iteration){
// extern "C" void jacobi_iterative(REAL * h_input, int width_y, int width_x, REAL * __var_0__, int iteration){
/* Host allocation Begin */
  host_printptx();
/*************************************/


//initialization
#if defined(PERSISTENT)
  auto execute_kernel = kernel_persistent_baseline<REAL>;
#endif
#if defined(BASELINE_CM)||defined(BASELINE)
  auto execute_kernel = kernel_baseline<REAL>;
#endif
#ifdef NAIVE
  auto execute_kernel = kernel2d_restrict<REAL>;
#endif 

  int sm_count;
  cudaDeviceGetAttribute ( &sm_count, cudaDevAttrMultiProcessorCount,0 );
  
  //initialization input and output space
  REAL * input;
  cudaMalloc(&input,sizeof(REAL)*((width_y-0)*(width_x-0)));
  Check_CUDA_Error("Allocation Error!! : input\n");
  cudaMemcpy(input,h_input,sizeof(REAL)*((width_y-0)*(width_x-0)), cudaMemcpyHostToDevice);
  REAL * __var_1__;
  cudaMalloc(&__var_1__,sizeof(REAL)*((width_y-0)*(width_x-0)));
  Check_CUDA_Error("Allocation Error!! : __var_1__\n");
  REAL * __var_2__;
  cudaMalloc(&__var_2__,sizeof(REAL)*((width_y-0)*(width_x-0)));
  Check_CUDA_Error("Allocation Error!! : __var_2__\n");

  //initialize tmp space for halo region
#if defined(GEN) || defined(MIX)|| defined(PERSISTENT)
  REAL * L2_cache3;
  REAL * L2_cache4;
  size_t L2_utage_2 = sizeof(REAL)*(width_y)*2*(width_x/bdim_x)*Halo;
#ifndef __PRINT__
  printf("l2 cache used is %ld KB : 4096 KB \n",L2_utage_2*2/1024);
#endif
  cudaMalloc(&L2_cache3,L2_utage_2*2);
  L2_cache4=L2_cache3+(width_y)*2*(width_x/bdim_x)*Halo;
#endif

  //initialize shared memory
  int maxSharedMemory;
  cudaDeviceGetAttribute (&maxSharedMemory, cudaDevAttrMaxSharedMemoryPerMultiprocessor,0 );
  //could not use all share memory in a100. so set it in default.
  int SharedMemoryUsed=maxSharedMemory-1024;

//#ifdef MIX
//  cudaFuncSetAttribute(kernel_mix<REAL>, cudaFuncAttributeMaxDynamicSharedMemorySize, SharedMemoryUsed);
//  cudaFuncSetAttribute(kernel_mix_reg<REAL>, cudaFuncAttributeMaxDynamicSharedMemorySize, SharedMemoryUsed);
//#endif
//#ifdef GEN
//  cudaFuncSetAttribute(kernel_general<REAL>, cudaFuncAttributeMaxDynamicSharedMemorySize, SharedMemoryUsed);
//#endif
//#ifdef PERSISTENT
//  cudaFuncSetAttribute(kernel_persistent_baseline<REAL>, cudaFuncAttributeMaxDynamicSharedMemorySize, SharedMemoryUsed);
//#endif
#if defined(USEMAXSM)
  cudaFuncSetAttribute(execute_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SharedMemoryUsed);
#endif

size_t executeSM=0;
#ifndef NAIVE
  //shared memory used for compuation
  size_t sharememory_basic=(1+(TILE_X+2*Halo)*(RTILE_Y+2*Halo))*sizeof(REAL);
  executeSM=sharememory_basic;
#endif
  #if defined(GEN) || defined(MIX)
  size_t sharememory3=sharememory2+(Halo*2*(TILE_Y))*sizeof(REAL);
  size_t sharememory4=sharememory3-(STILE_SIZE*sizeof(REAL));
#endif


#ifdef PERSISTENTTHREAD
  int numBlocksPerSm_current=0;
  #ifdef GEN
    if(SFOLDER_Y!=0)
    {
      // cudaLaunchCooperativeKernel((void*)kernel_general, grid_dim, block_dim, KernelArgs_NULL,sharememory3,0);
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksPerSm_current, kernel_general, bdim_x, sharememory3);
    }
    else
    {
      // cudaLaunchCooperativeKernel((void*)kernel_general, grid_dim, block_dim, KernelArgs_NULL,sharememory4,0);
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksPerSm_current, kernel_general, bdim_x, sharememory4);
    }
  
  #endif
  #ifdef MIX
    if(SFOLDER_Y!=0)
    {
      // cudaLaunchCooperativeKernel((void*)kernel_mix, grid_dim, block_dim, KernelArgs2,sharememory3,0);
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksPerSm_current, kernel_mix, bdim_x, sharememory3);
    }
    else
    {
      // cudaLaunchCooperativeKernel((void*)kernel_mix_reg, grid_dim, block_dim, KernelArgs2,sharememory4,0);
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksPerSm_current, kernel_mix_reg, bdim_x, sharememory4);
    }
  
  #endif
  #if defined(BASELINE_CM)||defined(PERSISTENT)
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksPerSm_current, execute_kernel, bdim_x, executeSM);
  #endif

  dim3 block_dim(bdim_x);
  dim3 grid_dim(width_x/bdim_x,sm_count*numBlocksPerSm_current/(width_x/bdim_x));
  
  dim3 executeBlockDim=block_dim;
  dim3 executeGridDim=grid_dim;
#endif 
#ifdef NAIVE
  dim3 block_dim_1(MIN(width_x,bdim_x),1);
  dim3 grid_dim_1(width_x/MIN(width_x,bdim_x),width_y/1);

  dim3 executeBlockDim=block_dim_1;
  dim3 executeGridDim=grid_dim_1;
#endif
#ifdef BASELINE
  dim3 block_dim2(bdim_x);
  dim3 grid_dim2(width_x/bdim_x,MIN((sm_count*8*1024/bdim_x)/(width_x/bdim_x),width_y/RTILE_Y));
  
  dim3 executeBlockDim=block_dim2;
  dim3 executeGridDim=grid_dim2;

#endif
//in order to get a better performance, warmup run is necessary.

#ifdef MIX
  int l_iteration=iteration;
  void* KernelArgs2[] ={(void**)&input,(void**)&width_y,
    (void*)&width_x,(void*)&__var_2__,(void*)&L2_cache1,(void*)&L2_cache1,
    (void*)&l_iteration};
#endif

#if defined(GEN) || defined(PERSISTENT)
  int l_iteration=iteration;
  void* KernelArgs4[] ={(void**)&input,(void**)&width_y,
    (void*)&width_x,(void*)&__var_2__,(void*)&L2_cache3,(void*)&L2_cache4,
    (void*)&l_iteration};
  void* ExecuteKernelArgs=KernelArgs4;
  #ifdef WARMUPRUN
    void* KernelArgs_NULL[] ={(void**)&__var_2__,(void**)&width_y,
      (void*)&width_x,(void*)&__var_1__,(void*)&L2_cache3,(void*)&L2_cache4,
      (void*)&l_iteration};
  #endif

#endif



#if defined(GEN) && defined(L2PER)
    REAL l2perused;
    size_t inner_window_size = 30*1024*1024;
    cudaStreamAttrValue stream_attribute;
    stream_attribute.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(L2_cache3);                  // Global Memory data pointer
    stream_attribute.accessPolicyWindow.num_bytes = min(inner_window_size,L2_utage_2*2);                                   // Number of bytes for persistence access
    stream_attribute.accessPolicyWindow.hitRatio  = 1;                                             // Hint for cache hit ratio
    stream_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;                  // Persistence Property
    stream_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;  

    cudaStreamSetAttribute(0, cudaStreamAttributeAccessPolicyWindow, &stream_attribute); 
    cudaCtxResetPersistingL2Cache();
    cudaStreamSynchronize(0);
#endif

#ifdef WARMUPRUN
  #ifdef TRADITIONLAUNCH
      execute_kernel<<<executeGridDim, executeBlockDim, executeSM>>>
            (__var_2__, width_y, width_x,__var_1__);
  #endif 

  #ifdef PERSISTENTLAUNCH
      cudaLaunchCooperativeKernel((void*)execute_kernel, executeGridDim, executeBlockDim, KernelArgs_NULL, executeSM,0);
  #endif

  #ifdef GEN
      printf("<%d,%d>\t<%d,%d>\t%d\t%d\t%d\t",bdim_x,1,grid_dim.x,grid_dim.y,grid_dim.x*grid_dim.y/sm_count,RFOLDER_Y,SFOLDER_Y);
      if(SFOLDER_Y!=0)
      {
        cudaLaunchCooperativeKernel((void*)kernel_general, grid_dim, block_dim, KernelArgs_NULL,sharememory3,0);
      }
      else
      {
        cudaLaunchCooperativeKernel((void*)kernel_general, grid_dim, block_dim, KernelArgs_NULL,sharememory4,0);
      }
  #endif
#endif 

#ifdef _TIMER_
  cudaEvent_t _forma_timer_start_,_forma_timer_stop_;
  cudaEventCreate(&_forma_timer_start_);
  cudaEventCreate(&_forma_timer_stop_);
  cudaEventRecord(_forma_timer_start_,0);
#endif


#ifdef MIX
  if(SFOLDER_Y!=0)
  {
    cudaLaunchCooperativeKernel((void*)kernel_mix, grid_dim, block_dim, KernelArgs2,sharememory3,0);
  }
  else
  {
    cudaLaunchCooperativeKernel((void*)kernel_mix_reg, grid_dim, block_dim, KernelArgs2,sharememory4,0);
  }
#endif

#ifdef GEN 
  if(SFOLDER_Y!=0)
  {
    cudaLaunchCooperativeKernel((void*)kernel_general, grid_dim, block_dim, KernelArgs4,sharememory3,0);
  }
  else
  {
    cudaLaunchCooperativeKernel((void*)kernel_general, grid_dim, block_dim, KernelArgs4,sharememory4,0);
  }
#endif

//#ifdef PERSISTENT
//  cudaLaunchCooperativeKernel((void*)kernel_persistent_baseline, grid_dim, block_dim, KernelArgs4, sharememory_basic,0);
//#endif
#ifdef PERSISTENTLAUNCH
  cudaLaunchCooperativeKernel((void*)execute_kernel, 
            executeGridDim, executeBlockDim, 
            //ExecuteKernelArgs, 
            KernelArgs4,
            executeSM,0);
#endif
#ifdef TRADITIONLAUNCH
  execute_kernel<<<executeGridDim, executeBlockDim, executeSM>>>
          (input, width_y, width_x, __var_2__);

  for(int i=1; i<iteration; i++)
  {
     execute_kernel<<<executeGridDim, executeBlockDim, executeSM>>>
          (__var_2__, width_y, width_x , __var_1__);
    REAL* tmp = __var_2__;
    __var_2__=__var_1__;
    __var_1__= tmp;
  }
#endif


#ifdef CHECK
  cudaDeviceSynchronize();
  cudaCheckError();
#endif

#ifndef __PRINT__  
  printf("sm_count is %d\n",sm_count);
  printf("MAX shared memory is %f KB but only use %f KB\n",maxSharedMemory/1024.0,SharedMemoryUsed/1024.0);
  size_t sharememory= (TILE_SM_X*TILE_SM_Y+1)*sizeof(REAL);
  printf(" shared meomory size is %ld KB\n", sharememory/1024);
  #if defined(BASELINE)||defined(BASELINE_CM)||defined(PERSISTENT)
    printf(" shared meomory size 0 (for computation and baseline) is %ld KB\n", sharememory_basic/1024);
  #endif
  #if defined(GEN) || defined(MIX)
    printf(" shared meomory size 3 (for general & mix)is %ld KB\n", sharememory3/1024);
  #endif
#endif

#ifdef __PRINT__
  #ifdef BASELINE
    #ifndef DA100X
      printf("bsln\t");
    #else
      printf("asyncbsln\t");
    #endif
  #endif 
  #ifdef BASELINE_CM
    #ifndef DA100X
      printf("bsln_cm\t");
    #else
      printf("asyncbsln_cm\t");
    #endif
  #endif 
  
  #ifdef NAIVE
    printf("naive\t");
  #endif 

  #ifdef PERSISTENT
    #ifndef DA100X
      printf("psstnt\t");
    #else
      printf("asyncpsstnt\t");
    #endif
  #endif

  // #ifdef GEN
  //     printf("gen"); 
  //   #else
  //     printf("asyncgen"); 
  //   #endif
  //   #if RFOLDER_Y==0 && SFOLDER_Y ==0
  //     printf("\t");
  //   #endif
  //   #if RFOLDER_Y==0 && SFOLDER_Y !=0
  //     printf("_sm\t");
  //   #endif
  //   #if RFOLDER_Y!=0 && SFOLDER_Y ==0
  //     printf("_reg\t");
  //   #endif
  //   #if RFOLDER_Y!=0 && SFOLDER_Y !=0
  //     printf("_mix\t");
  //   #endif
  // #endif
#endif 

#ifdef __PRINT__
  printf("%d\t%d\t%d\t",width_x,width_y,iteration);
  printf("<%d,%d>\t<%d,%d>\t%d\t0\t0\t",executeBlockDim.x,1,
        executeGridDim.x,executeGridDim.y,
        executeGridDim.x*executeGridDim.y/sm_count);
#endif

#ifdef _TIMER_
  cudaEventRecord(_forma_timer_stop_,0);
  cudaEventSynchronize(_forma_timer_stop_);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime,_forma_timer_start_,_forma_timer_stop_);
  #ifdef __PRINT__
  printf("%f\t%f\n",elapsedTime,(REAL)iteration*(width_y)*(width_x)/ elapsedTime/1000/1000);
  #else
  printf("[FORMA] Computation Time(ms) : %lf\n",elapsedTime);
  printf("[FORMA] Speed(GCells/s) : %lf\n",(REAL)iteration*(width_y)*(width_x)/ elapsedTime/1000/1000);
  printf("[FORMA] Speed(GFLOPS/s) : %lf\n", (REAL)17*iteration*(width_y)*(width_x)/ elapsedTime/1000/1000);
  printf("[FORMA] bandwidth(GB/s) : %lf\n", (REAL)sizeof(REAL)*iteration*((width_y)*(width_x)+width_x*width_y)/ elapsedTime/1000/1000);
  printf("[FORMA] width_x:width_y=%d:%d\n",(int)width_x, (int)width_y);
#if defined(GEN) || defined(PERSISTENT) || defined(MIX)
  printf("[FORMA] cached width_x:width_y=%d:%d\n",(int)TILE_X*grid_dim.x, (int)TILE_Y*grid_dim.y);
#endif
  printf("[FORMA] cached b:sf:rf=%d:%d:%d\n", (int)RTILE_Y, (int)SFOLDER_Y, (int)RFOLDER_Y);
  #endif

  cudaEventDestroy(_forma_timer_start_);
  cudaEventDestroy(_forma_timer_stop_);
#endif


//finalization
#ifdef CHECK
  // printf("check error here*\n");
  cudaDeviceSynchronize();
  cudaCheckError();
#endif

#if defined(GEN) || defined(PERSISTENT)
  if(iteration%2==1)
    cudaMemcpy(__var_0__,__var_2__, sizeof(REAL)*((width_y-0)*(width_x-0)), cudaMemcpyDeviceToHost);
  else
    cudaMemcpy(__var_0__,input, sizeof(REAL)*((width_y-0)*(width_x-0)), cudaMemcpyDeviceToHost);
#else
  cudaMemcpy(__var_0__,__var_2__, sizeof(REAL)*((width_y-0)*(width_x-0)), cudaMemcpyDeviceToHost);
#endif
/*Kernel Launch End */
/* Host Free Begin */
  cudaFree(input);
  cudaFree(__var_1__);
  cudaFree(__var_2__);

  // cudaFree(L2_cache);
  // cudaFree(L2_cache1);
  // cudaFree(L2_cache2);
#if defined(GEN) || defined(PERSISTENT)
  cudaFree(L2_cache3);
#endif
  // cudaFree(L2_cache4);

}

PERKS_INITIALIZE_ALL_TYPE(PERKS_DECLARE_INITIONIZATION_ITERATIVE);


