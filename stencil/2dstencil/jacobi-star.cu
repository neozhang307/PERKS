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

//direction of x axle is the same as thread index
//basic tiling: tiling unit for single thread
#ifndef RTILE_Y
#define RTILE_Y (8)
#endif
#ifndef TILE_X
#define TILE_X (256)
#endif

#define bdim_x (TILE_X)

#define BASIC_TILE_X (TILE_X+2*Halo)
#define BASIC_TILE_Y (RTILE_Y+2*Halo)
#define BASIC_SM_SPACE (BASIC_TILE_X)*(BASIC_TILE_Y)
/*********************ARGUMENTS for PERKS*******************************/
// Here "Folder" means how many times of "tiling unit" is stored in given memory structure
// Shared Memory folder of basic tiling
#ifndef SM_FOLER_Y
#define SM_FOLER_Y (2)
#endif
// Register Files folder of basic tiling
#ifndef REG_FOLER_Y
#define REG_FOLER_Y (6)
#endif
// Total 
#define TOTAL_SM_TILE_Y (RTILE_Y*SM_FOLER_Y)
#define TOTAL_REG_TILE_Y (RTILE_Y*REG_FOLER_Y)
#define TOTAL_SM_CACHE_SPACE (TILE_X+2*Halo)*(TOTAL_SM_TILE_Y+2*Halo)



#define TILE_Y (TOTAL_SM_TILE_Y+TOTAL_REG_TILE_Y)


//#define TILE_BASIC_TILE_X (TILE_X+Halo*2)
#define TILE_SM_Y (TOTAL_SM_TILE_Y+Halo*2)


#define BOULDER_STEP (Halo*2*(TILE_Y+TILE_X))
#define L2_STEP (BOULDER_STEP)
#define L2_EW_STEP (Halo*2*(TILE_Y)) 

#define E_STEP (0)
#define W_STEP (TILE_Y*Halo)
#define S_STEP (TILE_Y*Halo*2)
#define N_STEP (TILE_Y*Halo*2+Halo*TILE_X)

#define INITIAL (true)
#define NOTINITIAL (false)
#define SYNC (true)
#define NOSYNC (false)

//#undef TILE_Y

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
      sum[l_y]+=sm_ptr[(l_y+ps_y+sm_idx)*BASIC_TILE_X+local_x+ps_x-1-hl]*west[hl];\
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
      sum[l_y]+=sm_ptr[(l_y+ps_y+sm_idx)*BASIC_TILE_X+local_x+ps_x+1+hl]*east[hl];\
    }\
  }\
}while(0)

template<class REAL, int RESULT_SIZE, int halo, int INPUTREG_SIZE=(RESULT_SIZE+2*halo)>
__device__ void __forceinline__ computation(REAL result[RESULT_SIZE], 
                                            REAL* sm_ptr, int sm_y_base, int sm_x_ind,int sm_width, 
                                            REAL r_ptr[INPUTREG_SIZE], int reg_base, 
                                            const REAL west[6],const REAL east[6], 
                                            const REAL north[6],const REAL south[6],
                                            const REAL center 
                                          )
{
  _Pragma("unroll")
  for(int hl=0; hl<halo; hl++)
  {
    _Pragma("unroll")
    for(int l_y=0; l_y<RESULT_SIZE ; l_y++)
    {
      result[l_y]+=sm_ptr[(l_y+sm_y_base)*sm_width+sm_x_ind-1-hl]*west[hl];
    }
  }
  _Pragma("unroll")
  for(int hl=0; hl<halo; hl++)
  {
    _Pragma("unroll")
    for(int l_y=0; l_y<RESULT_SIZE ; l_y++)
    {
      result[l_y]+=r_ptr[reg_base+l_y-1-hl]*south[hl];
    }
  }
  _Pragma("unroll")
  for(int l_y=0; l_y<RESULT_SIZE ; l_y++)
  {
    result[l_y]+=r_ptr[reg_base+l_y]*center;
  }
  _Pragma("unroll")
  for(int hl=0; hl<halo; hl++)
  {
    _Pragma("unroll")
    for(int l_y=0; l_y<RESULT_SIZE ; l_y++)
    {
      result[l_y]+=r_ptr[reg_base+l_y+1+hl]*north[hl];
    }
  }
  _Pragma("unroll")
  for(int hl=0; hl<halo; hl++)
  {
    _Pragma("unroll")
    for(int l_y=0; l_y<RESULT_SIZE ; l_y++)
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


//store register array to global memory dst
template<class REAL, int SIZE, bool considerbound=true>
__device__ void __forceinline__ reg2global(REAL reg_array[SIZE], REAL* dst, 
  int global_y, int global_y_size, 
  int global_x, int global_x_size)
  {
    _Pragma("unroll")
    for(int l_y=0; l_y<SIZE; l_y++)
    {
      int l_global_y=global_y+l_y;
      if(considerbound==true)
      {
        if(l_global_y>=global_y_size|global_x>=global_x_size)
        {
          break;
        }
      }
      dst[(l_global_y) * global_x_size + global_x]=reg_array[l_y];
    }
}

template<class REAL, int SIZE, int halo>
__device__ void __forceinline__ global2reg(REAL*src, REAL reg_array[SIZE+2*halo],
  int global_y, int global_y_size,
  int global_x, int global_x_size)
{
  _Pragma("unroll")
  for (int l_y = 0; l_y < SIZE ; l_y++) 
  {
    {
      reg_array[l_y+halo] =  src[(l_y+global_y) * global_x_size + global_x];
    }
  }
}

template<class REAL, int START, int END>
__device__ void __forceinline__ ptrselfcp(REAL *ptr, 
                                      int ps_y, int y_step, 
                                      int local_x, int x_width)
{
  _Pragma("unroll")
  for(int l_y=START; l_y<END; l_y++)
  {
    int dst_ind=(l_y+ps_y)*(x_width);
    int src_ind=(l_y+ps_y+y_step)*(x_width);
    ptr[dst_ind+local_x]=ptr[src_ind+local_x];
    if(threadIdx.x<Halo*2)
        ptr[dst_ind+local_x+blockDim.x]=ptr[src_ind+local_x+blockDim.x];

  }
}

template<class REAL, int SRC_SIZE, int DST_SIZE, int SIZE>
__device__ void __forceinline__ reg2reg(REAL src_reg[SRC_SIZE], REAL dst_reg[DST_SIZE],
                                        int src_basic, int dst_basic)
{
  _Pragma("unroll")
  for(int l_y=0; l_y<SIZE; l_y++)
  {
    dst_reg[l_y+dst_basic]=src_reg[l_y+src_basic];
  }
}

//load global memory src to shared memory
template<class REAL, int START, int END, int halo, bool isInit=false, bool sync=true>
__device__ void __forceinline__ global2sm(REAL* src, REAL* sm_buffer, 
                                              int global_y_base, int global_y_size,
                                              int global_x_base, int global_x_size,
                                              int sm_y_base, int sm_x_base, int sm_width,
                                              int tid)
{
  if(START==END)return;
  //fill shared memory buffer
  _Pragma("unroll")
  for(int l_y=START; l_y<END; l_y++)
  {
    int l_global_y;
    if(isInit)
    {
      l_global_y=(MAX(global_y_base+l_y,0));
    }
    else
    {
      l_global_y=(MIN(global_y_base+l_y,global_y_size-1));
      l_global_y=(MAX(l_global_y,0));
    }
  
    #define  dst_ind (l_y+sm_y_base)*sm_width
    
    #ifndef ASYNCSM
      sm_buffer[dst_ind-halo+tid+sm_x_base]=src[l_global_y * global_x_size + MAX(global_x_base-halo+tid,0)];
      if(halo>0)
      {
        if(tid<halo*2)
        {  
          sm_buffer[dst_ind-halo+tid+blockDim.x+sm_x_base]=src[(l_global_y) * global_x_size + MIN(-halo+tid+blockDim.x+global_x_base, global_x_size-1)];
        }
      }
    #else
      __pipeline_memcpy_async(sm_buffer+dst_ind-halo+tid+ps_x, 
            src + (l_global_y) * global_x_size + MAX(glboal_x_base-halo+tid,0)
              , sizeof(REAL));
      if(halo>0)
      {
        if(tid<halo*2)
        {
          __pipeline_memcpy_async(sm_buffer+dst_ind-halo+tid+blockDim.x+ps_x, 
                  src + (l_global_y) * global_x_size + MIN(-halo+tid+blockDim.x+global_x_base,global_x_size-1)
                    , sizeof(REAL));
        }
      }
    #endif
  }
  #ifdef ASYNCSM
    if(sync=true)
    {
      __pipeline_commit();
      __pipeline_wait_prior(0);
    }
  #else
    __syncthreads();
  #endif

  #undef dst_ind
}

__device__ void __forceinline__ pipesync()
{
  #ifdef ASYNCSM
    {
      __pipeline_commit();
      __pipeline_wait_prior(0);
    }
  #else
    __syncthreads();
  #endif
}

template<class REAL, int SIZE, bool considerbound=true>
__device__ void __forceinline__ sm2global(REAL *sm_src, REAL* dst,
                                          int global_y_base, int global_y_size,
                                          int global_x_base, int global_x_size,
                                          int sm_y_base, int sm_x_base, int sm_width,
                                          int tid)
{

  _Pragma("unroll")
  for(int l_y=0; l_y<SIZE; l_y++)
  {
    int global_y=l_y+global_y_base;
    int global_x=tid+global_x_base;
    if(considerbound)
    {
      if(global_y>=global_y_size||global_x>=global_x_size)break;
    }
    dst[(global_y) * global_x_size + global_x] = sm_src[(sm_y_base + l_y) * sm_width + tid + sm_x_base];
  }
}

template<class REAL, int REG_SIZE, int SIZE>
__device__ void __forceinline__ sm2reg(REAL* sm_src, REAL reg_dst[SIZE],
                                      int y_base, 
                                      int x_base, int x_id,
                                      int sm_width, 
                                      int reg_base=0)
{
  _Pragma("unroll")
  for(int l_y=0; l_y<SIZE ; l_y++)
  {
    reg_dst[l_y+reg_base] = sm_src[(l_y+y_base)*sm_width+x_base+x_id];//input[(global_y) * width_x + global_x];
  }
}

template<class REAL, int REG_SIZE, int SIZE>
__device__ void __forceinline__ reg2sm( REAL reg_src[REG_SIZE], REAL* sm_dst,
                                      int sm_y_base, 
                                      int sm_x_base, int tid,
                                      int sm_width,
                                      int reg_base)
{
  _Pragma("unroll")
  for(int l_y=0; l_y<SIZE ; l_y++)
  {
    sm_dst[(l_y+sm_y_base)*sm_width + sm_x_base + tid]=reg_src[l_y+reg_base];
  }
}

#ifdef GEN
template<class REAL>
__global__ void kernel_general(REAL * __restrict__ input, int width_y, int width_x, 
  REAL * __restrict__ __var_4__, 
  REAL * __restrict__ l2_cache_o,REAL * __restrict__ l2_cache_i,
  int iteration)
{
  stencilParaT;
  //basic pointer
  cg::grid_group gg = cg::this_grid();
  //extern __shared__ REAL sm[];
  extern __shared__ char sm[];
  #if SM_FOLER_Y!=0
    //shared memory buffer space
    REAL* sm_space = (REAL*)sm+1;

    //shared memory buffer for register computation
    REAL* sm_rbuffer = sm_space + TOTAL_SM_CACHE_SPACE;
  #else
    REAL* sm_rbuffer = (REAL*)sm + 1;
  #endif

  register REAL* sm_rbuffers[2*Halo+RTILE_Y];
  sm_rbuffers[0]=sm_rbuffer;
  // _Pragma("unroll") 
  for(int lr=1; lr<2*Halo+RTILE_Y;lr++)
  {
    sm_rbuffers[lr]=sm_rbuffers[lr-1]+(bdim_x+2*Halo);
  }

  //boundary space
  REAL* boundary_buffer = sm_rbuffer + BASIC_SM_SPACE;

  //register buffer for shared memory c
  //register buffer space
  #if REG_FOLER_Y!=0
    register REAL r_space[REG_FOLER_Y*RTILE_Y+2*Halo];
  #endif
  register REAL r_smbuffer[2*Halo+RTILE_Y];

  const int tid = threadIdx.x;
  // int ps_x = Halo + tid;
  const int ps_y = Halo;
  const int ps_x = Halo;
  const int tile_basic_tile_x = blockDim.x + 2*Halo;

  const int p_x = blockIdx.x * TILE_X ;

  int blocksize_y=(width_y/gridDim.y);
  int y_quotient = width_y%gridDim.y;

  const int p_y =  blockIdx.y * (blocksize_y) + (blockIdx.y<=y_quotient?blockIdx.y:y_quotient);
  blocksize_y += (blockIdx.y<y_quotient?1:0);
  const int p_y_cache = p_y + (blocksize_y-TOTAL_REG_TILE_Y-TOTAL_SM_TILE_Y);

  //load data global to register
  // #pragma unroll
  #if REG_FOLER_Y !=0
    global2reg<REAL,TOTAL_REG_TILE_Y,Halo>(input, r_space,
                                              p_y_cache, width_y,
                                              p_x+tid, width_x);
  #endif
  // load data global to sm
  #if SM_FOLER_Y != 0
    global2sm<REAL,0,TOTAL_SM_TILE_Y,0>(input,sm_space,
                                        p_y_cache+TOTAL_REG_TILE_Y, width_y,
                                        p_x, width_x,
                                        ps_y, ps_x, BASIC_TILE_X,
                                        tid);
  #endif
  #if REG_FOLER_Y!=0 || SM_FOLER_Y!=0
    for(int local_y=tid; local_y<TILE_Y&&p_y_cache + local_y<width_y; local_y+=bdim_x)
    {
      for(int l_x=0; l_x<Halo; l_x++)
      {
        //east
        int global_x = p_x + TILE_X + l_x;
        global_x = MIN(width_x-1,global_x);
        boundary_buffer[E_STEP+local_y + l_x*TILE_Y] = input[(p_y_cache + local_y) * width_x + global_x];
        //west
        global_x = p_x - Halo + l_x;
        global_x = MAX(0,global_x);
        boundary_buffer[W_STEP+local_y + l_x*TILE_Y] =  input[(p_y_cache + local_y) * width_x + global_x];
      }
    }
    // sdfa
  #endif
  __syncthreads();


  for(int iter=0; iter<iteration; iter++)
  {
    int local_x=tid;
    //prefetch the boundary data
    //north south
    {
      //register
      #if REG_FOLER_Y!=0 || SM_FOLER_Y!=0
        // #pragma unroll
        _Pragma("unroll")
        for(int l_y=0; l_y<Halo; l_y++)
        {
          int global_y = (p_y_cache-Halo+l_y);
          global_y=MAX(0,global_y);
          //south
          #if REG_FOLER_Y != 0
            r_space[l_y]=input[(global_y) * width_x + p_x + tid];//boundary_buffer[S_STEP+tid + l_y*TILE_X];//sm_space[(ps_y+TOTAL_SM_TILE_Y-1) * BASIC_TILE_X + tid + ps_x];
          #else
            sm_space[(ps_y - Halo+l_y) * BASIC_TILE_X + tid + ps_x]=input[(global_y) * width_x + p_x + tid];
          #endif
        }
          //SM region
          // #pragma unroll
        _Pragma("unroll")
        for(int l_y=0; l_y<Halo; l_y++)
        {
          int global_y=(p_y_cache+(TOTAL_SM_TILE_Y+TOTAL_REG_TILE_Y)+l_y);
          global_y=MIN(global_y,width_y-1);
          //north
          #if SM_FOLER_Y != 0
            sm_space[(ps_y +TOTAL_SM_TILE_Y + l_y) * BASIC_TILE_X + tid + ps_x]=(input[(global_y) * width_x + p_x + tid]);//boundary_buffer[N_STEP+tid+l_y*TILE_X];
          #else
            r_space[TOTAL_REG_TILE_Y+Halo+l_y]=(input[(global_y) * width_x + p_x + tid]);
          #endif
        }
      #endif
      //NS of register & SM
      //*******************
      // #pragma unroll
      #if SM_FOLER_Y !=0 && REG_FOLER_Y !=0
        _Pragma("unroll")
        for(int l_y=0; l_y<Halo; l_y++)
        {
          //north of register
          r_space[TOTAL_REG_TILE_Y+Halo+l_y]=sm_space[(ps_y+l_y) * BASIC_TILE_X + tid + ps_x];//boundary_buffer[N_STEP+tid];
          //south of sm
          sm_space[(ps_y - Halo+l_y) * BASIC_TILE_X + tid + ps_x]=r_space[TOTAL_REG_TILE_Y+l_y];//boundary_buffer[S_STEP+tid];
        }
      #endif

    }

    //computation of general space 
    global2sm<REAL,-Halo,Halo,Halo,true,NOSYNC>(input, sm_rbuffer, 
                                            p_y, width_y,
                                            p_x, width_x,
                                            ps_y, ps_x, tile_basic_tile_x,
                                            tid);

    for(int global_y=p_y; global_y<p_y_cache; global_y+=RTILE_Y)
    {
      //data initialization
      // if(global_y==p_y)
      // {
      //   global2sm<REAL,-Halo,Halo,Halo,true,NOSYNC>(input, sm_rbuffer, 
      //                                       p_y, width_y,
      //                                       p_x, width_x,
      //                                       ps_y, ps_x, tile_basic_tile_x,
      //                                       tid);
      // }
      global2sm<REAL,Halo,RTILE_Y+Halo,Halo>(input, sm_rbuffer, 
                                          global_y, width_y,
                                          p_x, width_x,
                                          ps_y, ps_x, tile_basic_tile_x,
                                          tid);
      sm2reg<REAL,Halo*2+RTILE_Y, Halo*2+RTILE_Y>(sm_rbuffer, r_smbuffer, 
                                                    0,
                                                    ps_x, tid,
                                                    tile_basic_tile_x);
      REAL sum[RTILE_Y];
      init_reg_array<REAL,RTILE_Y>(sum,0);
      computation<REAL,RTILE_Y,Halo>(sum,
                                      sm_rbuffer, ps_y, local_x+ps_x, tile_basic_tile_x,
                                      r_smbuffer, Halo,
                                      west, east,
                                      north,south,  center);
      reg2global<REAL,RTILE_Y>(sum, __var_4__, 
                  global_y,p_y_cache, 
                  p_x+local_x, width_x);
      __syncthreads();
      ptrselfcp<REAL,-Halo, Halo>(sm_rbuffer, ps_y, RTILE_Y, tid, tile_basic_tile_x);
    }

    __syncthreads();
    //computation of register space
    // #pragma unroll
    #if REG_FOLER_Y !=0
      _Pragma("unroll")
      for(int local_y=0; local_y<TOTAL_REG_TILE_Y; local_y+=RTILE_Y)
      {
        //load data sm to buffer register
        //deal with ew boundary
        _Pragma("unroll")
        for(int l_y=tid; l_y<RTILE_Y; l_y+=bdim_x)
        {
          _Pragma("unroll")
          for(int l_x=0; l_x<Halo; l_x++)
          {
            // east
            sm_rbuffer[(l_y+ps_y)*BASIC_TILE_X+TILE_X + ps_x + l_x]=boundary_buffer[E_STEP+ l_y+local_y  + l_x*TILE_Y];
            // west
            sm_rbuffer[(l_y+ps_y)*BASIC_TILE_X+(-Halo) + ps_x + l_x]=boundary_buffer[W_STEP+l_y+local_y + l_x*TILE_Y];
          }
        }
        reg2sm<REAL, REG_FOLER_Y*RTILE_Y+2*Halo, RTILE_Y>(r_space, sm_rbuffer, 
                                                          ps_y, ps_x, tid, tile_basic_tile_x, local_y+Halo);
        __syncthreads();
        REAL sum[RTILE_Y];
        init_reg_array<REAL,RTILE_Y>(sum,0); 
        computation<REAL,RTILE_Y,Halo,REG_FOLER_Y*RTILE_Y+2*Halo>(sum,
                                      sm_rbuffer, ps_y, local_x+ps_x, tile_basic_tile_x,
                                      r_space, local_y+Halo,
                                      west, east,
                                      north,south,  center);
        reg2reg<REAL, RTILE_Y, REG_FOLER_Y*RTILE_Y+2*Halo, RTILE_Y>(sum,r_space,
                                      0, local_y);
        __syncthreads();
      }
    #endif
    #if SM_FOLER_Y != 0
      //computation of share memory space
      {
        //load shared memory boundary
        for(int local_y=tid; local_y<TOTAL_SM_TILE_Y; local_y+=bdim_x)
        {
          // _Pragma("unroll")
          for(int l_x=0; l_x<Halo; l_x++)
          {
            // east
            sm_space[(ps_y + local_y)*BASIC_TILE_X+TILE_X + ps_x+l_x]=boundary_buffer[E_STEP+local_y + TOTAL_REG_TILE_Y+ l_x*TILE_Y];
            //west
            sm_space[(ps_y + local_y)*BASIC_TILE_X+(-Halo) + ps_x+l_x]=boundary_buffer[W_STEP+local_y+ TOTAL_REG_TILE_Y+ l_x*TILE_Y];
          }
        }
        __syncthreads();
        //computation of shared space 
        sm2reg<REAL,RTILE_Y+2*Halo,2*Halo>(sm_space, r_smbuffer, 
                                            0, 
                                            ps_x, tid,
                                            tile_basic_tile_x,
                                            0);
        for ( size_t local_y = 0; local_y < TOTAL_SM_TILE_Y; local_y+=RTILE_Y) 
        {
          // if(local_y==0)
          // {
            
          // }
          sm2reg<REAL,RTILE_Y+2*Halo,RTILE_Y>(sm_space, r_smbuffer, 
                                            ps_y+local_y+Halo, 
                                            ps_x, tid,
                                            tile_basic_tile_x,
                                            Halo*2);
          REAL sum[RTILE_Y];
          init_reg_array<REAL,RTILE_Y>(sum,0);
          
          computation<REAL,RTILE_Y,Halo>(sum,
                                      sm_space, ps_y+local_y, local_x+ps_x, tile_basic_tile_x,
                                      r_smbuffer, Halo,
                                      west, east,
                                      north,south,  center);
          __syncthreads();
          reg2sm<REAL, RTILE_Y, RTILE_Y>(sum, sm_space,
                                      ps_y+local_y,
                                      ps_x, tid,
                                      tile_basic_tile_x,
                                      0);
          __syncthreads();
          reg2reg<REAL, RTILE_Y+2*Halo, RTILE_Y+2*Halo, 2*Halo>
                  (r_smbuffer,r_smbuffer,RTILE_Y, 0);
        }
      }
    #endif
    if(iter==iteration-1)break;
    //register memory related boundary
    //south
    //*******************
    #if REG_FOLER_Y!=0
      if(tid>=bdim_x-Halo)
      {
        int l_x=tid-bdim_x+Halo;
        //east
        // #pragma unroll
        _Pragma("unroll")
        for(int l_y=0; l_y<TOTAL_REG_TILE_Y; l_y++)
        {
          boundary_buffer[E_STEP + l_y + l_x*TILE_Y] = r_space[l_y];//sm_space[(ps_y + local_y) * BASIC_TILE_X + TILE_X + ps_x-Halo+0];
        }
      }
      else if(tid<Halo)
      {
        int l_x=tid;
        //west
        // #pragma unroll
        _Pragma("unroll")
        for(int l_y=0; l_y<TOTAL_REG_TILE_Y; l_y++)
        {
          boundary_buffer[W_STEP + l_y + l_x*TILE_Y] = r_space[l_y];//sm_space[(ps_y + local_y) * BASIC_TILE_X + TILE_X + ps_x-Halo+0];
        }
      }
    #endif
    // __syncthreads();
    //store sm related boundary
    // #pragma unroll
    #if SM_FOLER_Y != 0
      _Pragma("unroll")
      for(int local_y=tid; local_y<TOTAL_SM_TILE_Y; local_y+=bdim_x)
      {
        // #pragma unroll
        _Pragma("unroll")
        for(int l_x=0; l_x<Halo; l_x++)
        {
          //east
          boundary_buffer[E_STEP+local_y+TOTAL_REG_TILE_Y + l_x*TILE_Y] = sm_space[(ps_y + local_y) * BASIC_TILE_X + TILE_X + ps_x - Halo + l_x];
          //west
          boundary_buffer[W_STEP+local_y+TOTAL_REG_TILE_Y + l_x*TILE_Y] =  sm_space[(ps_y + local_y) * BASIC_TILE_X + ps_x + l_x];
        }
      }
    #endif
    //deal with sm related boundary
    //*******************
    //store boundary to global (NS)
    // #pragma unroll
    #if REG_FOLER_Y!=0 || SM_FOLER_Y!=0
      _Pragma("unroll")
      for(int l_y=0; l_y<Halo; l_y++)
      {
        //north
        #if SM_FOLER_Y!=0
          __var_4__[(p_y_cache+(TOTAL_SM_TILE_Y+TOTAL_REG_TILE_Y)-Halo+l_y) * width_x + p_x + tid]=sm_space[(ps_y + TOTAL_SM_TILE_Y - Halo+l_y) * BASIC_TILE_X + tid + ps_x];//boundary_buffer[N_STEP+tid+l_y*TILE_X];//
        #else
          __var_4__[(p_y_cache+(TOTAL_SM_TILE_Y+TOTAL_REG_TILE_Y)-Halo+l_y) * width_x + p_x + tid]=r_space[l_y+TOTAL_REG_TILE_Y-Halo];
        #endif

         //south
        #if REG_FOLER_Y!=0
          __var_4__[(p_y_cache+l_y) * width_x + p_x + tid]= r_space[l_y];//boundary_buffer[S_STEP+tid+l_y*TILE_X] ;//
        #else
          __var_4__[(p_y_cache+l_y) * width_x + p_x + tid]= sm_space[(ps_y + l_y) * BASIC_TILE_X + tid + ps_x];//r_space[l_y];
        #endif
      }
    #endif
    //*******************
    //store register part boundary
    __syncthreads();
    // store the whole boundary space to l2 cache
    
    // #pragma unroll
    #if REG_FOLER_Y!=0 || SM_FOLER_Y!=0
      _Pragma("unroll")
      for(int lid=tid; lid<TILE_Y; lid+=bdim_x)
      {
        //east
        // #pragma unroll
        _Pragma("unroll")
        for(int l_x=0; l_x<Halo; l_x++)
        {
           //east
          l2_cache_o[(((blockIdx.x* 2 +1)* Halo+l_x)*width_y)  + p_y_cache +lid] = boundary_buffer[E_STEP+lid +l_x*TILE_Y];
          //west
          l2_cache_o[(((blockIdx.x* 2 + 0) * Halo+l_x)*width_y)  + p_y_cache +lid] = boundary_buffer[W_STEP+lid+l_x*TILE_Y];
        }
      }
    #endif
    gg.sync();

    REAL* tmp_ptr=__var_4__;
    __var_4__=input;
    input=tmp_ptr;

    #if REG_FOLER_Y!=0 || SM_FOLER_Y!=0
      tmp_ptr=l2_cache_o;
      l2_cache_o=l2_cache_i;
      l2_cache_i=tmp_ptr;
    
       // #pragma unroll
      _Pragma("unroll")
      for(int local_y=tid; local_y<TILE_Y; local_y+=bdim_x)
      {
        #pragma unroll
        for(int l_x=0; l_x<Halo; l_x++)
        {
          //east
           boundary_buffer[E_STEP+local_y+l_x*TILE_Y] = ((blockIdx.x == gridDim.x-1)?boundary_buffer[E_STEP+local_y+(Halo-1)*TILE_Y]:
             l2_cache_i[(((blockIdx.x+1)*2+0)* Halo+l_x)*width_y + p_y_cache + local_y]);

           //west
           boundary_buffer[W_STEP+local_y+l_x*TILE_Y] = ((blockIdx.x == 0)?boundary_buffer[W_STEP+local_y+0*TILE_Y]:
            l2_cache_i[(((blockIdx.x-1)*2+1)* Halo+l_x)*width_y + p_y_cache + local_y]);
        }
      }
    #endif

    #if REG_FOLER_Y !=0

      _Pragma("unroll")
      for(int l_y=TOTAL_REG_TILE_Y-1; l_y>=0; l_y--)
      {
        r_space[l_y+Halo]=r_space[l_y];
      }
    #endif

  }
  #if REG_FOLER_Y!=0
    // register->global
    reg2global<REAL, TOTAL_REG_TILE_Y, false>(r_space, __var_4__,
                                      p_y_cache, width_y,
                                      p_x+tid, width_x);
  #endif
  
  #if SM_FOLER_Y!=0
    __syncthreads();
    // shared memory -> global
    sm2global<REAL,TOTAL_SM_TILE_Y,false>(sm_space, __var_4__,
                                    p_y_cache+TOTAL_REG_TILE_Y, width_y,
                                    p_x, width_x,
                                    ps_y, ps_x, BASIC_TILE_X,
                                    tid);
  #endif
}
#endif

#if defined(BASELINE_CM)||defined(BASELINE)||defined(PERSISTENT)

template<class REAL, int LOCAL_TILE_Y>
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

  register REAL r_smbuffer[2*Halo+LOCAL_TILE_Y];

  const int tid = threadIdx.x;
  // int ps_x = Halo + tid;
  const int ps_y = Halo;
  const int ps_x = Halo;
   // REAL* sb2 = sb+TILE_BASIC_TILE_X*TILE_SM_Y;
  const int p_x = blockIdx.x * blockDim.x;
  const int tile_basic_tile_x = blockDim.x + 2*Halo;

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

    global2sm<REAL,-Halo,Halo,Halo,true,NOSYNC>(input, sm_rbuffer, 
                                            p_y, width_y,
                                            p_x, width_x,
                                            ps_y, ps_x, tile_basic_tile_x,
                                            tid);
                                            

    //computation of register space 
    for(int global_y=p_y; global_y<p_y_end; global_y+=LOCAL_TILE_Y)
    {

      global2sm<REAL,Halo,LOCAL_TILE_Y+Halo,Halo>(input, sm_rbuffer, 
                                            global_y, width_y,
                                            p_x, width_x,
                                            ps_y, ps_x, tile_basic_tile_x,
                                            tid);

      //__syncthreads();
      //shared memory buffer -> register buffer
      sm2reg<REAL,Halo*2+LOCAL_TILE_Y, Halo*2+LOCAL_TILE_Y>(sm_rbuffer, r_smbuffer, 
                                                    0,
                                                    ps_x, tid,
                                                    tile_basic_tile_x);

      REAL sum[LOCAL_TILE_Y];
      init_reg_array<REAL,LOCAL_TILE_Y>(sum,0);
      //main computation
      //COMPUTE2(sm_rbuffer,0,r_smbuffer,0);
      
      computation<REAL,LOCAL_TILE_Y,Halo>(sum,
                                      sm_rbuffer, ps_y, local_x+ps_x, tile_basic_tile_x,
                                      r_smbuffer, Halo,
                                      west, east,
                                      north,south,  center);

      //store to global
      reg2global<REAL,LOCAL_TILE_Y>(sum, __var_4__, 
                  global_y,p_y_end, 
                  p_x+local_x, width_x);
      __syncthreads();
      
      //some data in shared memroy can be used in next tiling. 
      ptrselfcp<REAL,-Halo, Halo>(sm_rbuffer, ps_y, LOCAL_TILE_Y, tid, tile_basic_tile_x);

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
  auto execute_kernel = kernel_persistent_baseline<REAL,RTILE_Y>;
#endif
#if defined(BASELINE_CM)||defined(BASELINE)
  auto execute_kernel = kernel_baseline<REAL,RTILE_Y>;
#endif
#ifdef NAIVE
  auto execute_kernel = kernel2d_restrict<REAL>;
#endif 
#ifdef GEN
  auto execute_kernel = kernel_general<REAL>;
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
  size_t sharememory_basic=(1+BASIC_SM_SPACE)*sizeof(REAL);
  executeSM=sharememory_basic;
#endif
  #if defined(GEN) || defined(MIX)
  size_t sm_cache_size = TOTAL_SM_CACHE_SPACE*sizeof(REAL);
  size_t y_axle_halo = (Halo*2*TILE_Y)*sizeof(REAL);

  executeSM=sharememory_basic+y_axle_halo;
  if(SM_FOLER_Y!=0)executeSM+=sm_cache_size;
  //size_t sharememory3=sharememory_basic+(Halo*2*(TILE_Y))*sizeof(REAL);
  //size_t sharememory4=sharememory3-(STILE_SIZE*sizeof(REAL));
#endif


#ifdef PERSISTENTTHREAD
  int numBlocksPerSm_current=0;
  #ifdef GEN
    // if(SM_FOLER_Y!=0)
    // {
    //   // cudaLaunchCooperativeKernel((void*)kernel_general, grid_dim, block_dim, KernelArgs_NULL,sharememory3,0);
    //   cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    //     &numBlocksPerSm_current, kernel_general, bdim_x, sharememory3);
    // }
    // else
    // {
    //   // cudaLaunchCooperativeKernel((void*)kernel_general, grid_dim, block_dim, KernelArgs_NULL,sharememory4,0);
    //   cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    //     &numBlocksPerSm_current, kernel_general, bdim_x, sharememory4);
    // }
  
  #endif
  #ifdef MIX
    if(SM_FOLER_Y!=0)
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
  #if defined(BASELINE_CM)||defined(PERSISTENT)||defined(GEN)
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
  void* ExecuteKernelArgs[] ={(void**)&input,(void**)&width_y,
    (void*)&width_x,(void*)&__var_2__,(void*)&L2_cache3,(void*)&L2_cache4,
    (void*)&l_iteration};

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

#endif 

#ifdef _TIMER_
  cudaEvent_t _forma_timer_start_,_forma_timer_stop_;
  cudaEventCreate(&_forma_timer_start_);
  cudaEventCreate(&_forma_timer_stop_);
  cudaEventRecord(_forma_timer_start_,0);
#endif


#ifdef MIX
  if(SM_FOLER_Y!=0)
  {
    cudaLaunchCooperativeKernel((void*)kernel_mix, grid_dim, block_dim, KernelArgs2,sharememory3,0);
  }
  else
  {
    cudaLaunchCooperativeKernel((void*)kernel_mix_reg, grid_dim, block_dim, KernelArgs2,sharememory4,0);
  }
#endif

//
#ifdef PERSISTENTLAUNCH
  cudaLaunchCooperativeKernel((void*)execute_kernel, 
            executeGridDim, executeBlockDim, 
            ExecuteKernelArgs, 
            //KernelArgs4,
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
  printf(" shared meomory size is %ld KB\n", executeSM/1024);

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
  //   #if REG_FOLER_Y==0 && SM_FOLER_Y ==0
  //     printf("\t");
  //   #endif
  //   #if REG_FOLER_Y==0 && SM_FOLER_Y !=0
  //     printf("_sm\t");
  //   #endif
  //   #if REG_FOLER_Y!=0 && SM_FOLER_Y ==0
  //     printf("_reg\t");
  //   #endif
  //   #if REG_FOLER_Y!=0 && SM_FOLER_Y !=0
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
  printf("[FORMA] cached b:sf:rf=%d:%d:%d\n", (int)RTILE_Y, (int)SM_FOLER_Y, (int)REG_FOLER_Y);
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


