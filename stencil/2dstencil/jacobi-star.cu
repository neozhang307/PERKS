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

/*********************ARGUMENTS for PERKS*******************************/
// Here "Folder" means how many times of "tiling unit" is stored in given memory structure
// Shared Memory folder of basic tiling
// #ifndef SM_FOLER_Y
// #define SM_FOLER_Y (2)
// #endif
// Register Files folder of basic tiling
#ifndef REG_FOLDER_Y
#define REG_FOLDER_Y (6)
#endif
// Total 
#define ISINITI (true)
#define NOTINITIAL (false)
#define SYNC (true)
#define NOSYNC (false)

//#undef TILE_Y
// #define USESM

#ifdef USESM
  #define USESMSET (true)
#else
  #define USESMSET (false)
#endif


#define cudaCheckError() {                                          \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
 }                                                                 \
}

#ifndef BOX
#define stencilParaT \
  const REAL west[6]={12.0/118,9.0/118,3.0/118,2.0/118,5.0/118,6.0/118};\
  const REAL east[6]={12.0/118,9.0/118,3.0/118,3.0/118,4.0/118,6.0/118};\
  const REAL north[6]={5.0/118,7.0/118,5.0/118,4.0/118,3.0/118,2.0/118};\
  const REAL south[6]={5.0/118,7.0/118,5.0/118,1.0/118,6.0/118,2.0/118};\
  const REAL center=15.0/118;
;
  #define stencilParaList const REAL west[6],const REAL east[6],const REAL north[6],const REAL south[6],const REAL center
  #define stencilParaInput  west,east,north,south,center
  #define R_PTR r_ptr[INPUTREG_SIZE]
#else
  #if Halo==1
  #define stencilParaT \
  const REAL filter[3][3] = {\
    {7.0/118, 5.0/118, 9.0/118},\
    {12.0/118,15.0/118,12.0/118},\
    {9.0/118, 5.0/118, 7.0/118}\
  };
  #endif
  #if Halo==2
  #define stencilParaT \
  const REAL filter[5][5] = {\
    {1.0/118, 2.0/118, 3.0/118, 4.0/118, 5.0/118},\
    {7.0/118, 7.0/118, 5.0/118, 7.0/118, 6.0/118},\
    {8.0/118,12.0/118,15.0/118,12.0/118,12.0/118},\
    {9.0/118, 9.0/118, 5.0/118, 7.0/118, 15.0/118},\
    {10.0/118, 11.0/118, 12.0/118, 13.0/118, 14.0/118}\
  };
  #endif

  #define stencilParaList const REAL filter[halo*2+1][halo*2+1]
  #define stencilParaInput  filter
  #define R_PTR r_ptr[2*halo+1][INPUTREG_SIZE]
#endif


template<class REAL, int RESULT_SIZE, int halo, int INPUTREG_SIZE=(RESULT_SIZE+2*halo)>
__device__ void __forceinline__ computation(REAL result[RESULT_SIZE], 
                                            REAL* sm_ptr, int sm_y_base, int sm_x_ind,int sm_width, 
                                            REAL R_PTR,
                                            int reg_base, 
                                            stencilParaList
                                            // const REAL west[6],const REAL east[6], 
                                            // const REAL north[6],const REAL south[6],
                                            // const REAL center 
                                          )
{
#ifndef BOX
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
#else
  _Pragma("unroll")\
  for(int hl_y=-halo; hl_y<=halo; hl_y++)
  {
    _Pragma("unroll")
    for(int hl_x=-halo; hl_x<=halo; hl_x++)
    {
      _Pragma("unroll")
      for(int l_y=0; l_y<RESULT_SIZE ; l_y++)
      {
        result[l_y]+=filter[hl_y+halo][hl_x+halo]*r_ptr[hl_x+halo][hl_y+halo+l_y];
      }
    }
  }
#endif
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


template<class REAL, int SIZE_REG, int SIZE, bool considerbound=true>
__device__ void __forceinline__ reg2global(REAL reg_array[SIZE_REG], REAL* dst, 
  int global_y, int global_y_size, 
  int global_x, int global_x_size,
  int reg_base=0)
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
      dst[(l_global_y) * global_x_size + global_x]=reg_array[l_y+reg_base];
    }
}



template<class REAL, int REG_SIZE, int SIZE>
__device__ void __forceinline__ global2reg(REAL*src, REAL reg_array[REG_SIZE],
  int global_y, int global_y_size,
  int global_x, int global_x_size,
  int reg_base)
{
  _Pragma("unroll")
  for (int l_y = 0; l_y < SIZE ; l_y++) 
  {
    {
      reg_array[l_y+reg_base] =  src[(l_y+global_y) * global_x_size + global_x];
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

template<class REAL, int SRC_SIZE, int DST_SIZE, int SIZE,int halo=0>
__device__ void __forceinline__ reg2reg(REAL src_reg[SRC_SIZE], REAL dst_reg[DST_SIZE],
                                        int src_basic, int dst_basic)
{
  _Pragma("unroll")
  for(int l_y=0; l_y<SIZE; l_y++)
  {
    dst_reg[l_y+dst_basic]=src_reg[l_y+src_basic];
  }
}

template<class REAL, int SRC_SIZE, int DST_SIZE, int SIZE, int halo>
__device__ void __forceinline__ regs2regs(REAL src_reg[2*halo+1][SRC_SIZE], REAL dst_reg[2*halo+1][DST_SIZE],
                                        int src_basic, int dst_basic)
{
  _Pragma("unroll")
  for(int l_x=0; l_x<halo*2+1; l_x++)
  {
    _Pragma("unroll")
    for(int l_y=0; l_y<SIZE; l_y++)
    {
      dst_reg[l_x][l_y+dst_basic]=src_reg[l_x][l_y+src_basic];
    }
  }
}

template<class REAL, int halo, bool isInit=false, bool sync=true>
__device__ void __forceinline__ global2sm(REAL* src, REAL* sm_buffer, 
                                              int size, 
                                              int global_y_base, int global_y_size,
                                              int global_x_base, int global_x_size,
                                              int sm_y_base, int sm_x_base, int sm_width,
                                              int tid)
{
  //fill shared memory buffer
  _Pragma("unroll")
  for(int l_y=0; l_y<size; l_y++)
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
  if(sync==true)
  {  
    #ifdef ASYNCSM

      __pipeline_commit();
      __pipeline_wait_prior(0);
    #else
      __syncthreads();
    #endif
  }
  
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

//template<class REAL, int SIZE, bool considerbound=true>
template<class REAL, bool considerbound=true>
__device__ void __forceinline__ sm2global(REAL *sm_src, REAL* dst,
                                          int size, 
                                          int global_y_base, int global_y_size,
                                          int global_x_base, int global_x_size,
                                          int sm_y_base, int sm_x_base, int sm_width,
                                          int tid)
{

 // _Pragma("unroll")
  for(int l_y=0; l_y<size; l_y++)
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

template<class REAL, int REG_SIZE, int SIZE, int halo=0>
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

template<class REAL, int REG_SIZE, int SIZE, int halo>
__device__ void __forceinline__ sm2regs(REAL* sm_src, REAL reg_dst[2*halo+1][REG_SIZE],
                                      int y_base, 
                                      int x_base, int x_id,
                                      int sm_width, 
                                      int reg_base=0)
{
  _Pragma("unroll")
  for(int l_x=0; l_x<halo*2+1; l_x++)
  {
    _Pragma("unroll")
    for(int l_y=0; l_y<SIZE ; l_y++)
    {
      reg_dst[l_x][l_y+reg_base] = sm_src[(l_y+y_base)*sm_width+x_base+x_id+l_x-halo];//input[(global_y) * width_x + global_x];
    }
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
template<class REAL, int LOCAL_TILE_Y=RTILE_Y, int halo=Halo, int reg_folder_y=REG_FOLDER_Y, bool UseSMCache=USESMSET>
__global__ void kernel_general(REAL * __restrict__ input, int width_y, int width_x, 
  REAL * __restrict__ __var_4__, 
  REAL * __restrict__ l2_cache_o,REAL * __restrict__ l2_cache_i,
  int iteration,
  int max_sm_flder)
{
  if(!UseSMCache) max_sm_flder=0;
  #define UseRegCache (reg_folder_y!=0)
  #ifdef BOX
    #define SM2REG sm2regs
    #define REG2REG regs2regs
  #else
    #define SM2REG sm2reg
    #define REG2REG reg2reg
  #endif
  stencilParaT;
  //basic pointer
  cg::grid_group gg = cg::this_grid();
  //extern __shared__ REAL sm[];
  extern __shared__ char sm[];
 
  const int total_sm_tile_y = LOCAL_TILE_Y*max_sm_flder;//SM_FOLER_Y;//consider how to automatically compute it later
  const int total_reg_tile_y = LOCAL_TILE_Y*reg_folder_y;
  const int total_tile_y = total_sm_tile_y+total_reg_tile_y;
  const int total_reg_tile_y_with_halo = total_reg_tile_y+2*halo;

  const int sizeof_rspace = total_reg_tile_y_with_halo;
  const int sizeof_rbuffer = LOCAL_TILE_Y+2*halo;

  const int tile_x = blockDim.x;
  const int tile_x_with_halo = tile_x + 2*halo;
  const int tile_y_with_halo = LOCAL_TILE_Y+2*halo;
  const int basic_sm_space=tile_x_with_halo*tile_y_with_halo;

  const int boundary_line_size = total_tile_y;
  const int e_step = 0;
  const int w_step = boundary_line_size*halo;

  REAL* sm_rbuffer =(REAL*)sm+1;

  REAL* boundary_buffer = sm_rbuffer + basic_sm_space;
  REAL* sm_space = boundary_buffer+(2*halo*boundary_line_size);//BOX need add additional stuffs. 


  //boundary space
  //register buffer space
  //seems use much space than necessary when no use register version. 
  register REAL r_space[total_reg_tile_y_with_halo];
#ifndef BOX
  register REAL r_smbuffer[2*halo+LOCAL_TILE_Y];
#else
  register REAL r_smbuffer[2*halo+1][2*halo+LOCAL_TILE_Y];
#endif

  const int tid = threadIdx.x;
  // int ps_x = Halo + tid;
  const int ps_y = halo;
  const int ps_x = halo;
 // const int tile_x_with_halo = blockDim.x + 2*halo;

  const int p_x = blockIdx.x * tile_x ;

  int blocksize_y=(width_y/gridDim.y);
  int y_quotient = width_y%gridDim.y;
  
  const int p_y =  blockIdx.y * (blocksize_y) + (blockIdx.y<=y_quotient?blockIdx.y:y_quotient);
  blocksize_y += (blockIdx.y<y_quotient?1:0);
  const int p_y_cache = p_y + (blocksize_y-total_reg_tile_y-total_sm_tile_y);

  //load data global to register
  // #pragma unroll
  if(UseRegCache)
  {
    global2reg<REAL,sizeof_rspace,total_reg_tile_y>(input, r_space,
                                              p_y_cache, width_y,
                                              p_x+tid, width_x,
                                              halo);
  }
  // load data global to sm
  if(UseSMCache)
  {
    global2sm<REAL,0>(input,sm_space,
                                        total_sm_tile_y,
                                        p_y_cache+total_reg_tile_y, width_y,
                                        p_x, width_x,
                                        ps_y, ps_x, tile_x_with_halo,
                                        tid);
  }
  //load ew boundary
  if(UseRegCache||UseSMCache)
  {
    for(int local_y=tid; local_y<boundary_line_size&&p_y_cache + local_y<width_y; local_y+=blockDim.x)
    {
      for(int l_x=0; l_x<halo; l_x++)
      {
        //east
        int global_x = p_x + tile_x + l_x;
        global_x = MIN(width_x-1,global_x);
        boundary_buffer[e_step+local_y + l_x*boundary_line_size] = input[(p_y_cache + local_y) * width_x + global_x];
        //west
        global_x = p_x - halo + l_x;
        global_x = MAX(0,global_x);
        boundary_buffer[w_step+local_y + l_x*boundary_line_size] =  input[(p_y_cache + local_y) * width_x + global_x];
      }
    }
    // sdfa
  }
  __syncthreads();

  for(int iter=0; iter<iteration; iter++)
  {
    int local_x=tid;
    //prefetch the boundary data
    //north south
    {
      //register
      if(UseRegCache||UseSMCache)
      {  // #pragma unroll
        _Pragma("unroll")
        for(int l_y=0; l_y<halo; l_y++)
        {
          int global_y = (p_y_cache-halo+l_y);
          global_y=MAX(0,global_y);
          //south
          if(UseRegCache)
          {
            r_space[l_y]=input[(global_y) * width_x + p_x + tid];
          }
          else
          {
            sm_space[(ps_y - halo + l_y) * tile_x_with_halo + tid + ps_x]=input[(global_y) * width_x + p_x + tid];
          }
          global_y=(p_y_cache+(total_sm_tile_y+total_reg_tile_y)+l_y);
          global_y=MIN(global_y,width_y-1);
          //north
          if(UseSMCache)
          {
            sm_space[(ps_y +total_sm_tile_y + l_y) * tile_x_with_halo + tid + ps_x]=(input[(global_y) * width_x + p_x + tid]);
          }
          else
          {
            r_space[total_reg_tile_y+halo+l_y]=(input[(global_y) * width_x + p_x + tid]);
          }
          if(UseRegCache && UseSMCache)
          {
            //north of register
            r_space[total_reg_tile_y+halo+l_y]=sm_space[(ps_y+l_y) * tile_x_with_halo + tid + ps_x];
            //south of sm
            sm_space[(ps_y - halo+l_y) * tile_x_with_halo + tid + ps_x]=r_space[total_reg_tile_y+l_y];
          }
        }
      }
    }

    //computation of general space 
    global2sm<REAL,halo,ISINITI,NOSYNC>(input, sm_rbuffer, 
                                            halo*2,
                                            p_y-halo, width_y,
                                            p_x, width_x,
                                            ps_y-halo, ps_x, tile_x_with_halo,
                                            tid);

    for(int global_y=p_y; global_y<p_y_cache; global_y+=LOCAL_TILE_Y)
    {

      global2sm<REAL,halo>(input, sm_rbuffer,
                                          LOCAL_TILE_Y, 
                                          global_y+halo, width_y,
                                          p_x, width_x,
                                          ps_y+halo, ps_x, tile_x_with_halo,
                                          tid);
      sm2reg<REAL,sizeof_rbuffer, sizeof_rbuffer>(sm_rbuffer, r_smbuffer, 
                                                    0,
                                                    ps_x, tid,
                                                    tile_x_with_halo);
      REAL sum[LOCAL_TILE_Y];
      init_reg_array<REAL,LOCAL_TILE_Y>(sum,0);
      computation<REAL,LOCAL_TILE_Y,halo>(sum,
                                      sm_rbuffer, ps_y, local_x+ps_x, tile_x_with_halo,
                                      r_smbuffer, halo,
                                      stencilParaInput);
      reg2global<REAL,LOCAL_TILE_Y,LOCAL_TILE_Y>(sum, __var_4__, 
                  global_y,p_y_cache, 
                  p_x+local_x, width_x);
      __syncthreads();
      ptrselfcp<REAL,-halo, halo>(sm_rbuffer, ps_y, LOCAL_TILE_Y, tid, tile_x_with_halo);
    }

    __syncthreads();
    //computation of register space

    if(UseRegCache)
    {
      _Pragma("unroll")
      for(int local_y=0; local_y<total_reg_tile_y; local_y+=LOCAL_TILE_Y)
      {
        //load data sm to buffer register
        //deal with ew boundary
        _Pragma("unroll")
        for(int l_y=tid; l_y<LOCAL_TILE_Y; l_y+=blockDim.x)
        {
          _Pragma("unroll")
          for(int l_x=0; l_x<halo; l_x++)
          {
            // east
            sm_rbuffer[(l_y+ps_y)*tile_x_with_halo+ tile_x + ps_x + l_x]=boundary_buffer[e_step + l_y + local_y + l_x * boundary_line_size];
            // west
            sm_rbuffer[(l_y+ps_y)*tile_x_with_halo+(-halo) + ps_x + l_x]=boundary_buffer[w_step + l_y + local_y + l_x * boundary_line_size];
          }
        }
        reg2sm<REAL, sizeof_rspace, LOCAL_TILE_Y>(r_space, sm_rbuffer, 
                                                          ps_y, ps_x, tid, tile_x_with_halo, local_y+halo);
        
        
        __syncthreads();
        REAL sum[LOCAL_TILE_Y];
        init_reg_array<REAL,LOCAL_TILE_Y>(sum,0); 
        computation<REAL,LOCAL_TILE_Y,halo,sizeof_rspace>(sum,
                                      sm_rbuffer, ps_y, local_x+ps_x, tile_x_with_halo,
                                      r_space, local_y+halo,
                                      stencilParaInput);
        reg2reg<REAL, LOCAL_TILE_Y, sizeof_rspace, LOCAL_TILE_Y>(sum,r_space,
                                      0, local_y);
        __syncthreads();
      }
    }
    if(UseSMCache)
    //computation of share memory space
    {
      //load shared memory boundary
      for(int local_y=tid; local_y<total_sm_tile_y; local_y+=blockDim.x)
      {
        // _Pragma("unroll")
        for(int l_x=0; l_x<halo; l_x++)
        {
          // east
          sm_space[(ps_y + local_y)*tile_x_with_halo+ tile_x + ps_x+l_x] = boundary_buffer[e_step + local_y + total_reg_tile_y + l_x*boundary_line_size];
          //west
          sm_space[(ps_y + local_y)*tile_x_with_halo+(-halo) + ps_x+l_x] = boundary_buffer[w_step + local_y + total_reg_tile_y + l_x*boundary_line_size];
        }
      }
      __syncthreads();
      //computation of shared space 
      sm2reg<REAL,sizeof_rbuffer,2*halo>(sm_space, r_smbuffer, 
                                          0, 
                                          ps_x, tid,
                                          tile_x_with_halo,
                                          0);
      for ( size_t local_y = 0; local_y < total_sm_tile_y; local_y+=LOCAL_TILE_Y) 
      {
        sm2reg<REAL,sizeof_rbuffer,LOCAL_TILE_Y>(sm_space, r_smbuffer, 
                                          ps_y+local_y+halo, 
                                          ps_x, tid,
                                          tile_x_with_halo,
                                          halo*2);
        REAL sum[LOCAL_TILE_Y];
        init_reg_array<REAL,LOCAL_TILE_Y>(sum,0);
        
        computation<REAL,LOCAL_TILE_Y,halo>(sum,
                                    sm_space, ps_y+local_y, local_x+ps_x, tile_x_with_halo,
                                    r_smbuffer, halo,
                                    stencilParaInput);
        __syncthreads();
        reg2sm<REAL, LOCAL_TILE_Y, LOCAL_TILE_Y>(sum, sm_space,
                                    ps_y+local_y,
                                    ps_x, tid,
                                    tile_x_with_halo,
                                    0);
        __syncthreads();
        reg2reg<REAL, sizeof_rbuffer, sizeof_rbuffer, 2*halo>
                (r_smbuffer,r_smbuffer, LOCAL_TILE_Y, 0);
      }
    }
    
    if(iter==iteration-1)break;
    //register memory related boundary
    //south
    //*******************
    if(UseRegCache)
    {
      if(tid>=blockDim.x-halo)
      {
        int l_x=tid-blockDim.x+halo;
        //east
        // #pragma unroll
        _Pragma("unroll")
        for(int l_y=0; l_y<total_reg_tile_y; l_y++)
        {
          boundary_buffer[e_step + l_y + l_x*boundary_line_size] = r_space[l_y];//sm_space[(ps_y + local_y) * BASIC_TILE_X + TILE_X + ps_x-Halo+0];
        }
      }
      else if(tid<halo)
      {
        int l_x=tid;
        //west
        // #pragma unroll
        _Pragma("unroll")
        for(int l_y=0; l_y<total_reg_tile_y; l_y++)
        {
          boundary_buffer[w_step + l_y + l_x*boundary_line_size] = r_space[l_y];//sm_space[(ps_y + local_y) * BASIC_TILE_X + TILE_X + ps_x-Halo+0];
        }
      }
    }
    //store sm related boundary
    if(UseSMCache)
    {
      _Pragma("unroll")
      for(int local_y=tid; local_y<total_sm_tile_y; local_y+=blockDim.x)
      {
        _Pragma("unroll")
        for(int l_x=0; l_x<halo; l_x++)
        {
          //east
          boundary_buffer[e_step+local_y+total_reg_tile_y + l_x*boundary_line_size] = sm_space[(ps_y + local_y) * tile_x_with_halo + tile_x + ps_x - halo + l_x];
          //west
          boundary_buffer[w_step+local_y+total_reg_tile_y + l_x*boundary_line_size] = sm_space[(ps_y + local_y) * tile_x_with_halo + ps_x + l_x];
        }
      }
    }
    //deal with sm related boundary
    //*******************
    //store boundary to global (NS)
    if(UseRegCache||UseSMCache)
    {
      _Pragma("unroll")
      for(int l_y=0; l_y<halo; l_y++)
      {
        //north
        if(UseSMCache)
        {
          __var_4__[(p_y_cache+(total_sm_tile_y+total_reg_tile_y)-halo+l_y) * width_x + p_x + tid]=sm_space[(ps_y + total_sm_tile_y - halo+l_y) * tile_x_with_halo + tid + ps_x];//boundary_buffer[N_STEP+tid+l_y*TILE_X];//
        }
        else
        {
          __var_4__[(p_y_cache+(total_sm_tile_y+total_reg_tile_y)-halo+l_y) * width_x + p_x + tid]=r_space[l_y+total_reg_tile_y-halo];
        }
         //south
        if(UseRegCache)
        {
          __var_4__[(p_y_cache+l_y) * width_x + p_x + tid]= r_space[l_y];
        }
        else
        {
          __var_4__[(p_y_cache+l_y) * width_x + p_x + tid]= sm_space[(ps_y + l_y) * tile_x_with_halo + tid + ps_x];
        }
      }
    }
    //*******************
    //store register part boundary
    __syncthreads();
    // store the whole boundary space to l2 cache
    if(UseSMCache||UseRegCache)
    {
      _Pragma("unroll")
      for(int lid=tid; lid<boundary_line_size; lid+=blockDim.x)
      {
        //east
        _Pragma("unroll")
        for(int l_x=0; l_x<halo; l_x++)
        {
           //east
          l2_cache_o[(((blockIdx.x* 2 + 1 )* halo+l_x)*width_y)  + p_y_cache +lid] = boundary_buffer[e_step+lid +l_x*boundary_line_size];
          //west
          l2_cache_o[(((blockIdx.x* 2 + 0) * halo+l_x)*width_y)  + p_y_cache +lid] = boundary_buffer[w_step+lid+l_x*boundary_line_size];
        }
      }
    }
    gg.sync();

    REAL* tmp_ptr=__var_4__;
    __var_4__=input;
    input=tmp_ptr;

    if(UseRegCache||UseSMCache)
    {
      tmp_ptr=l2_cache_o;
      l2_cache_o=l2_cache_i;
      l2_cache_i=tmp_ptr;
    
      _Pragma("unroll")
      for(int local_y=tid; local_y<boundary_line_size; local_y+=blockDim.x)
      {
        _Pragma("unroll")
        for(int l_x=0; l_x<halo; l_x++)
        {
          //east
           boundary_buffer[e_step+local_y+l_x*boundary_line_size] = ((blockIdx.x == gridDim.x-1)?boundary_buffer[e_step+local_y+(halo-1)*boundary_line_size]:
             l2_cache_i[(((blockIdx.x+1)*2+0)* halo+l_x)*width_y + p_y_cache + local_y]);
           //west
           boundary_buffer[w_step+local_y+l_x*boundary_line_size] = ((blockIdx.x == 0)?boundary_buffer[w_step+local_y+0*boundary_line_size]:
            l2_cache_i[(((blockIdx.x-1)*2+1)* halo+l_x)*width_y + p_y_cache + local_y]);
        }
      }
    }

    if(UseRegCache)
    {
      _Pragma("unroll")
      for(int l_y=total_reg_tile_y-1; l_y>=0; l_y--)
      {
        r_space[l_y+halo]=r_space[l_y];
      }
    }

  }

  if(UseRegCache)
  {
    // register->global
    reg2global<REAL, sizeof_rspace, total_reg_tile_y, false>(r_space, __var_4__,
                                      p_y_cache, width_y,
                                      p_x+tid, width_x,
                                      0);
  }
  
  if(UseSMCache)
  {
    __syncthreads();
    // shared memory -> global
    sm2global<REAL,false>(sm_space, __var_4__, 
                                    total_sm_tile_y,
                                    p_y_cache+total_reg_tile_y, width_y,
                                    p_x, width_x,
                                    ps_y, ps_x, tile_x_with_halo ,
                                    tid);
  }
  #undef UseRegCache
  #undef SM2REG
  #undef REG2REG
}
#endif

#if defined(BASELINE_CM)||defined(BASELINE)||defined(PERSISTENT)

template<class REAL, int LOCAL_TILE_Y, int halo=Halo>
#ifdef PERSISTENT
__global__ void kernel_persistent_baseline(REAL * __restrict__ input, int width_y, int width_x, 
  REAL * __restrict__ __var_4__,REAL * __restrict__ l2_cache=NULL, REAL * __restrict__ l2_cachetmp=NULL, 
  int iteration=0)
#else
__global__ void kernel_baseline(REAL * __restrict__ input, int width_y, int width_x, 
  REAL * __restrict__ __var_4__)
#endif

{

  #ifdef BOX
    #define SM2REG sm2regs
    #define REG2REG regs2regs
    #define isBOX (halo)
  #else
    #define SM2REG sm2reg
    #define REG2REG reg2reg
    #define isBOX (0)
  #endif

  stencilParaT;
  extern __shared__ char sm[];
  
  REAL* sm_space = (REAL*)sm+1;
  REAL* sm_rbuffer = sm_space;

#ifndef BOX
  register REAL r_smbuffer[2*halo+LOCAL_TILE_Y];
#else
  register REAL r_smbuffer[2*halo+1][2*halo+LOCAL_TILE_Y];
#endif

  const int tid = threadIdx.x;
  // int ps_x = Halo + tid;
  const int ps_y = halo;
  const int ps_x = halo;
   // REAL* sb2 = sb+TILE_BASIC_TILE_X*TILE_SM_Y;
  const int p_x = blockIdx.x * blockDim.x;
  const int tile_x_with_halo = blockDim.x + 2*halo;

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

    global2sm<REAL,halo,ISINITI,NOSYNC>(input, sm_rbuffer, 
                                            2*halo,
                                            p_y-halo, width_y,
                                            p_x, width_x,
                                            ps_y-halo, ps_x, tile_x_with_halo,
                                            tid);
                                            

    //computation of register space 
    for(int global_y=p_y; global_y<p_y_end; global_y+=LOCAL_TILE_Y)
    {

      global2sm<REAL,halo>(input, sm_rbuffer,
                                            LOCAL_TILE_Y, 
                                            global_y+halo, width_y,
                                            p_x, width_x,
                                            ps_y+halo, ps_x, tile_x_with_halo,
                                            tid);

      //__syncthreads();
      //shared memory buffer -> register buffer
      SM2REG<REAL,halo*2+LOCAL_TILE_Y, halo*2+LOCAL_TILE_Y,isBOX>(sm_rbuffer, r_smbuffer, 
                                                    0,
                                                    ps_x, tid,
                                                    tile_x_with_halo);

      REAL sum[LOCAL_TILE_Y];
      init_reg_array<REAL,LOCAL_TILE_Y>(sum,0);
      //main computation
      //COMPUTE2(sm_rbuffer,0,r_smbuffer,0);
      
      computation<REAL,LOCAL_TILE_Y,halo>(sum,
                                      sm_rbuffer, ps_y, local_x+ps_x, tile_x_with_halo,
                                      r_smbuffer, halo,
                                      stencilParaInput);

      //store to global
      reg2global<REAL,LOCAL_TILE_Y,LOCAL_TILE_Y>(sum, __var_4__, 
                  global_y,p_y_end, 
                  p_x+local_x, width_x);
      __syncthreads();
      
      //some data in shared memroy can be used in next tiling. 
      ptrselfcp<REAL,-halo, halo>(sm_rbuffer, ps_y, LOCAL_TILE_Y, tid, tile_x_with_halo);

    }

    #ifdef PERSISTENT
      if(iter==iteration-1)break;
  
      gg.sync();

      REAL* tmp_ptr =__var_4__;
      __var_4__=input;
      input=tmp_ptr;
    #endif
  }
  #undef SM2REG
  #undef REG2REG
  #undef isBOX
} 
#endif

#ifdef NAIVE
template<class REAL, int halo=Halo>
__global__ void kernel2d_restrict(REAL* input,
                                  int width_y, int width_x, REAL* output) 
{
  stencilParaT;
  int l_x = blockDim.x * blockIdx.x + threadIdx.x;  
  int l_y = blockDim.y * blockIdx.y + threadIdx.y;

#ifndef BOX
  int c = l_x + l_y * width_x;
  int w[halo];
  int e[halo];
  int n[halo];
  int s[halo];
  _Pragma("unroll") 
  for(int hl=0; hl<halo; hl++)
  {
    w[hl] = MAX(0,l_x-1-hl)+l_y * width_x;
    e[hl] = MIN(width_x-1,l_x+1+hl)+l_y * width_x;
    s[hl] = l_x+MAX(0,l_y-1-hl) * width_x;;
    n[hl] = l_x+MIN(width_y-1,l_y+1+hl) * width_x;
  }
  REAL sum=0;
  _Pragma("unroll") 
  for(int hl=0; hl<halo; hl++)
  {
    sum+=south[hl]*input[s[hl]];
  }
  _Pragma("unroll") 
  for(int hl=0; hl<halo; hl++)
  {
    sum+=west[hl]*input[w[hl]];
  }
  sum+=center*input[c];
  _Pragma("unroll") 
  for(int hl=0; hl<halo; hl++)
  {
    sum+=east[hl]*input[e[hl]];
  }
  _Pragma("unroll") 
  for(int hl=0; hl<halo; hl++)
  {
    sum+=north[hl]*input[n[hl]];
  }
  output[c]=sum;
  return;
#else
  int vertical[Halo*2+1];
  int horizontal[Halo*2+1];
  #pragma unroll
  for(int hl_y=-Halo; hl_y<=Halo; hl_y++)
  {
    vertical[hl_y+Halo]=MIN(MAX(l_y+hl_y,0),width_y-1)*width_x;
  }
  #pragma unroll
  for(int hl_x=-Halo; hl_x<=Halo; hl_x++)
  {
    horizontal[hl_x+Halo]=MIN(MAX(l_x+hl_x,0),width_x-1);
  }
  REAL sum=0;
  #pragma unroll
  for(int hl_y=-Halo; hl_y<=Halo; hl_y++)
  {
    #pragma unroll 
    for(int hl_x=-Halo; hl_x<=Halo; hl_x++)
    {
      sum+=filter[hl_y+Halo][hl_x+Halo]*input[vertical[hl_y+Halo]  + horizontal[hl_x+Halo]];
    }
  }
  output[vertical[Halo]  + horizontal[Halo]]=sum;
  return;
#endif
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


#define TOTAL_SM_TILE_Y (RTILE_Y*SM_FOLER_Y)
#define TOTAL_REG_TILE_Y (RTILE_Y*REG_FOLDER_Y)
#define TOTAL_SM_CACHE_SPACE (TILE_X+2*Halo)*(TOTAL_SM_TILE_Y+2*Halo)

#define TILE_Y (TOTAL_SM_TILE_Y+TOTAL_REG_TILE_Y)




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
  int SharedMemoryUsed = maxSharedMemory-1024;

#if defined(USEMAXSM)
  cudaFuncSetAttribute(execute_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SharedMemoryUsed);
#endif

size_t executeSM = 0;
size_t max_sm_flder=0;
#ifndef NAIVE
  //shared memory used for compuation
  int basic_sm_space=(RTILE_Y+2*Halo)*(TILE_X+2*Halo);
  size_t sharememory_basic=(1+basic_sm_space)*sizeof(REAL);
  executeSM = sharememory_basic;
#endif
  #if defined(GEN) || defined(MIX)
  max_sm_flder=(SharedMemoryUsed/sizeof(REAL)
                          -basic_sm_space
                          -2*Halo*REG_FOLDER_Y*RTILE_Y
                          -2*Halo*(TILE_X+2*Halo))/(TILE_X+4*Halo)/RTILE_Y;

  // size_t sm_cache_size = TOTAL_SM_CACHE_SPACE*sizeof(REAL);
  size_t sm_cache_size = (max_sm_flder*RTILE_Y+2*Halo)*(TILE_X+2*Halo)*sizeof(REAL);
  size_t y_axle_halo = (Halo*2*(max_sm_flder+REG_FOLDER_Y)*RTILE_Y)*sizeof(REAL);
  executeSM=sharememory_basic+y_axle_halo;
  executeSM+=sm_cache_size;
#ifndef __PRINT__
  printf("the max flder is %ld and the total sm size is %ld\n", max_sm_flder, executeSM);
#endif

  //size_t sharememory3=sharememory_basic+(Halo*2*(TILE_Y))*sizeof(REAL);
  //size_t sharememory4=sharememory3-(STILE_SIZE*sizeof(REAL));
#endif


#ifdef PERSISTENTTHREAD
  int numBlocksPerSm_current=0;

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
    (void*)&l_iteration, (void*)&max_sm_flder};

  #ifdef WARMUPRUN
    void* KernelArgs_NULL[] ={(void**)&__var_2__,(void**)&width_y,
      (void*)&width_x,(void*)&__var_1__,(void*)&L2_cache3,(void*)&L2_cache4,
      (void*)&l_iteration, (void *)&max_sm_flder};
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
  //   #if REG_FOLDER_Y==0 && SM_FOLER_Y ==0
  //     printf("\t");
  //   #endif
  //   #if REG_FOLDER_Y==0 && SM_FOLER_Y !=0
  //     printf("_sm\t");
  //   #endif
  //   #if REG_FOLDER_Y!=0 && SM_FOLER_Y ==0
  //     printf("_reg\t");
  //   #endif
  //   #if REG_FOLDER_Y!=0 && SM_FOLER_Y !=0
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
  printf("[FORMA] cached width_x:width_y=%d:%d\n",(int)TILE_X*grid_dim.x, (int)(max_sm_flder+REG_FOLDER_Y)*RTILE_Y*grid_dim.y);
#endif
  printf("[FORMA] cached b:sf:rf=%d:%d:%d\n", (int)RTILE_Y, (int)max_sm_flder, (int)REG_FOLDER_Y);
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


