#ifndef CONFIGURE
  #include "./config.cuh"
#endif
#include "./common/cuda_computation.cuh"
#include "./common/cuda_common.cuh"
#include "./common/types.hpp"
#include <math.h>

#include <cooperative_groups.h>

#ifdef SMASYNC
  #if PERKS_ARCH<800 
    #error "unsupport architecture"
  #endif
  #include <cooperative_groups/memcpy_async.h>
  #include <cuda_pipeline.h>
#endif

namespace cg = cooperative_groups;


// #if defined(BASELINE_CM)||defined(BASELINE)||defined(PERSISTENT)

template<class REAL, int LOCAL_TILE_Y, int halo>
#ifdef PERSISTENT
__global__ void 
#ifndef BOX
kernel_persistent_baseline
#else
kernel_persistent_baseline_box
#endif
(REAL * __restrict__ input, int width_y, int width_x, 
  REAL *__restrict__ __var_4__,REAL *__restrict__ l2_cache, REAL *__restrict__ l2_cachetmp, 
  int iteration)
#else
__global__ void 
#ifndef BOX
kernel_baseline
#else
kernel_baseline_box
#endif
(REAL * __restrict__ input, int width_y, int width_x, 
  REAL * __restrict__ __var_4__)
#endif

{

  #ifdef BOX
    #define SM2REG sm2regs
    #define REG2REG regs2regs
  #else
    #define SM2REG sm2reg
    #define REG2REG reg2reg
  #endif

  stencilParaT;
  extern __shared__ char sm[];
  
  REAL* sm_space = (REAL*)sm+1;
  REAL* sm_rbuffer = sm_space;
  // __shared__ REAL sm_rbuffer[(RTILE_Y+2*HALO)*(TILE_X+2*HALO)+1];
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
  const int sizeof_rbuffer=2*halo+RTILE_Y; 
  // for(int iter=0; iter<iteration; iter++)
#ifdef PERSISTENT  
  cg::grid_group gg = cg::this_grid();
  for(int iter=0; iter<iteration; iter++)
#endif
  {
    int local_x=tid;

    global2sm<REAL,halo,ISINITI,SYNC>(input, sm_rbuffer, 
                                            2*halo,
                                            p_y-halo, width_y,
                                            p_x, width_x,
                                            ps_y-halo, ps_x, tile_x_with_halo,
                                            tid);
                                            
    SM2REG<REAL,sizeof_rbuffer, halo*2,isBOX>(sm_rbuffer, r_smbuffer, 
                                                    0,
                                                    ps_x, tid,
                                                    tile_x_with_halo);
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
      SM2REG<REAL,sizeof_rbuffer, LOCAL_TILE_Y,isBOX>(sm_rbuffer, r_smbuffer, 
                                                   2*halo,
                                                    ps_x, tid,
                                                    tile_x_with_halo,
                                                    2*halo); 

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
      ptrselfcp<REAL,-halo, halo,halo>(sm_rbuffer, ps_y, LOCAL_TILE_Y, tid, tile_x_with_halo);
      REG2REG<REAL, sizeof_rbuffer, sizeof_rbuffer, 2*halo,isBOX>
                (r_smbuffer,r_smbuffer, LOCAL_TILE_Y, 0);
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
// #endiif
#ifndef CONFIGURE
  #ifndef BOX
      #ifndef PERSISTENT
          PERKS_INITIALIZE_ALL_TYPE_2ARG(PERKS_DECLARE_INITIONIZATION_BASELINE,RTILE_Y,HALO);
      #else
          PERKS_INITIALIZE_ALL_TYPE_2ARG(PERKS_DECLARE_INITIONIZATION_PBASELINE,RTILE_Y,HALO);
      #endif
  #else
      #ifndef PERSISTENT
          PERKS_INITIALIZE_ALL_TYPE_2ARG(PERKS_DECLARE_INITIONIZATION_BASELINE_BOX,RTILE_Y,HALO);
      #else
          PERKS_INITIALIZE_ALL_TYPE_2ARG(PERKS_DECLARE_INITIONIZATION_PBASELINE_BOX,RTILE_Y,HALO);
      #endif
  #endif
#else
  #ifndef BOX
      #ifdef PERSISTENT
          template __global__ void kernel_persistent_baseline<TYPE,RTILE_Y,HALO>(TYPE*__restrict__,int,int,TYPE*__restrict__,TYPE*__restrict__,TYPE*__restrict__,int );
      #else
          template __global__ void kernel_baseline<TYPE,RTILE_Y,HALO>(TYPE*__restrict__,int,int,TYPE*__restrict__,TYPE*__restrict__,TYPE*__restrict__,int );
      #endif
  #else
      #ifdef PERSISTENT
          template __global__ void kernel_persistent_baseline_box<TYPE,RTILE_Y,HALO>(TYPE*__restrict__,int,int,TYPE*__restrict__,TYPE*__restrict__,TYPE*__restrict__,int );
      #else
          template __global__ void kernel_baseline_box<TYPE,RTILE_Y,HALO>(TYPE*__restrict__,int,int,TYPE*__restrict__,TYPE*__restrict__,TYPE*__restrict__,int );
      #endif
  #endif
#endif
// template __global__ void kernel_baseline<float, RTILE_Y,HALO>(float*,int,int,float*);
// template void kernel_baseline<double, HALO>(double*,int,int,double*);