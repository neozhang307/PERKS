#include "./config.cuh"

#include "./common/cuda_computation.cuh"
#include "./common/cuda_common.cuh"
#include "./common/types.hpp"
#include <math.h>

#include <cooperative_groups.h>

#ifdef ASYNCSM
  // #if PERKS_ARCH<800 
    // #error "unsupport architecture"
  // #endif
  #include <cooperative_groups/memcpy_async.h>
  #include <cuda_pipeline.h>
#endif

namespace cg = cooperative_groups;

template<class REAL, int halo>
__global__ void 
#ifndef PERSISTENT
kernel3d_baseline(REAL * __restrict__ input, 
                                REAL * __restrict__ output, 
                                int width_z, int width_y, int width_x) 
#else
kernel3d_persistent(REAL * __restrict__ input, 
                                REAL * __restrict__ output, 
                                int width_z, int width_y, int width_x,
                                REAL* l2_cache_i, REAL* l2_cache_o,
                                int iteration) 
#endif
{
  // printf("?");
  const int tile_x_with_halo=TILE_X+2*halo;
  const int tile_y_with_halo=TILE_Y+2*halo;
  stencilParaT;
  
  extern __shared__ char sm[];
  REAL* sm_rbuffer = (REAL*)sm+1;

  register REAL r_smbuffer[2*halo+1][ITEM_PER_THREAD];
  // printf("%d\n",ITEM_PER_THREAD);
  // return;
  REAL* smbuffer_buffer_ptr[halo+1];
  smbuffer_buffer_ptr[0]=sm_rbuffer;
  #pragma unroll
  for(int hl=1; hl<halo+1; hl++)
  {
    smbuffer_buffer_ptr[hl]=smbuffer_buffer_ptr[hl-1]+tile_x_with_halo*tile_y_with_halo;
  }

  const int tid_x = threadIdx.x%TILE_X;
  const int tid_y = threadIdx.x/TILE_X;
  const int dim_y = TILE_Y/ITEM_PER_THREAD;

  const int cpblocksize_y=(TILE_Y+2*halo)/dim_y;
  const int cpquotion_y=(TILE_Y+2*halo)%dim_y;

  const int index_y = ITEM_PER_THREAD*tid_y;

  const int cpbase_y = -halo+tid_y*cpblocksize_y+(tid_y<=cpquotion_y?tid_y:cpquotion_y);
  const int cpend_y = cpbase_y + cpblocksize_y + (tid_y<=cpquotion_y?1:0);
  // if(tid_x==0&&blockIdx.x==0)
  //   printf("<%d,%d,%d,%d,%d,%d>",cpbase_y,cpend_y,cpblocksize_y,dim_y,tid_y,ITEM_PER_THREAD);
  // return;
  const int ps_y = halo;
  const int ps_x = halo;
  // const int ps_z = halo;

  const int p_x = blockIdx.x * TILE_X;
  const int p_y = blockIdx.y * TILE_Y;
  // if(blockIdx.x==0&&tid_x==0)
  //   printf("<%d,%d,%d:%d,%d>",ps_x,ps_y,ps_z,tid_x,tid_y);
  // return;
  int blocksize_z=(width_z/gridDim.z);
  int z_quotient = width_z%gridDim.z;

  const int p_z =  blockIdx.z * (blocksize_z) + (blockIdx.z<=z_quotient?blockIdx.z:z_quotient);
  blocksize_z += (blockIdx.z<z_quotient?1:0);
  const int p_z_end = p_z + (blocksize_z);
 
  // int smz_ind=0;
#ifdef PERSISTENT  
  cg::grid_group gg = cg::this_grid();
  for(int iter=0; iter<iteration; iter++)
#endif
  {
    global2regs3d<REAL, ITEM_PER_THREAD, 1+2*halo>
      (input, r_smbuffer, p_z-halo,width_z, p_y+index_y, width_y, p_x, width_x,tid_x);
    //need optimization to remove redundent memory access
    global2sm<REAL, halo, 0, halo, halo+1, 0, true, false>
                                        (input, smbuffer_buffer_ptr,
                                          p_x, p_y, p_z,
                                          width_x, width_y, width_z,

                                          tile_x_with_halo, ps_x,
                                          // -halo+index_y, -halo+index_y+ITEM_PER_THREAD+2*halo, ps_y,
                                          cpbase_y, cpend_y,ps_y,
                                          // cpsize_y, ps_y,cpbase_y, 
                                          TILE_X, tid_x);
    __syncthreads();
    for(int global_z=p_z; global_z<p_z_end; global_z+=1)
    {
      __syncthreads();
      
      global2sm<REAL, halo, halo, 1, halo+1, 0, false, true>
                                        (input, smbuffer_buffer_ptr,
                                          p_x, p_y, global_z,
                                          width_x, width_y, width_z,

                                          tile_x_with_halo, ps_x,
                                          // -halo+index_y, -halo+index_y+ITEM_PER_THREAD+2*halo, ps_y,
                                          cpbase_y, cpend_y,ps_y,
                                          TILE_X,tid_x);
      __syncthreads();
      

      //sm2reg

      sm2regs<REAL, ITEM_PER_THREAD, 1+2*halo, 
                1+halo, halo, 
                0, halo*2, 
                ITEM_PER_THREAD, 1>
        (smbuffer_buffer_ptr, r_smbuffer, 
          ps_y+index_y, ps_x, 
          tile_x_with_halo, tid_x);

      REAL sum[ITEM_PER_THREAD];
      // #pragma unroll
      _Pragma("unroll")
      for(int l_y=0; l_y<ITEM_PER_THREAD; l_y++)
      {
        sum[l_y]=0;
      }

      //main computation
      computation<REAL,ITEM_PER_THREAD,halo>( sum,
                                      smbuffer_buffer_ptr[0],
                                      ps_y+index_y, tile_x_with_halo, tid_x+ps_x,
                                      r_smbuffer,
                                      stencilParaInput);

      // // reg 2 ptr
      reg2global3d<REAL, ITEM_PER_THREAD>(
            sum, output,
            global_z, width_z,
            p_y+index_y, width_y,
            p_x, width_x,
            tid_x);

      REAL* tmp = smbuffer_buffer_ptr[0];
      // smswap 
      _Pragma("unroll")
      for(int hl=1; hl<halo+1; hl++)
      {
        smbuffer_buffer_ptr[hl-1]=smbuffer_buffer_ptr[hl];
      }
      smbuffer_buffer_ptr[halo]=tmp;

      regsself3d<REAL,2*halo+1,ITEM_PER_THREAD>(r_smbuffer);

    }
    #ifdef PERSISTENT
      if(iter>=iteration-1)break;
      // if(threadIdx.x==0&&blockIdx.x==0&&blockIdx.y==0&&blockDim.x==0)
      // {
      //   printf("%d\n",iter);
      // }
      gg.sync();

      REAL* tmp_ptr =output;
      output=input;
      input=tmp_ptr;
    #endif
  }
}


// template __global__ void kernel3d_baseline<float,HALO> 
//     (float *__restrict__, float *__restrict__ , int , int , int );
// template __global__ void kernel3d_baseline<double,HALO> 
//     (double *__restrict__, double *__restrict__ , int , int , int );

#ifndef PERSISTENT 
  PERKS_INITIALIZE_ALL_TYPE_1ARG(PERKS_DECLARE_INITIONIZATION_BASELINE,HALO);
#else
  PERKS_INITIALIZE_ALL_TYPE_1ARG(PERKS_DECLARE_INITIONIZATION_PERSISTENT,HALO);
#endif