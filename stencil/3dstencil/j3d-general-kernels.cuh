#include "./config.cuh"
#include "./genconfig.cuh"
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

template<class REAL, int halo, int LOCAL_ITEM_PER_THREAD, int LOCAL_TILE_X, int LOCAL_TILE_Y, int LOCAL_NOCACHE_Y, int SM_SIZE_Z, int REG_SIZE_Z,
          int REG_CACHESIZE_Z=1,          
          bool loadfrmcache=false, bool storetocache=false,
          bool isloadfrmreg=false, bool isstoretoreg=false>
__device__ void __forceinline__ process_one_layer 
      (
        REAL * __restrict__ input, REAL * __restrict__ output, 
        REAL* smbuffer_buffer_ptr[SM_SIZE_Z], 
        REAL r_smbuffer[REG_SIZE_Z][LOCAL_ITEM_PER_THREAD],

        int global_z, int p_y, int p_x,
        int width_z, int width_y, int width_x,

        int ps_y, int ps_x, int sm_width_x,

        int cpbase_y, int cpend_y, int index_y,

        int tid_x, int tid_y,
        stencilParaList,
        REAL* sm_space=NULL, int frmcachesmid_z=0, int tocachesmid_z=0,
        REAL  r_space[REG_CACHESIZE_Z][LOCAL_ITEM_PER_THREAD]=NULL, int frmcacheregid_z=0, int tocacheregid_z=0,
        REAL* boundary_buffer=NULL, int boundary_buffer_index_z=0, 
        const int boundary_east_step=0, const int boundary_west_step=0,
        const int boundary_step_yz=0
        
      )
{
  // __syncthreads();
  //in(global, halo+glboal_z)
  if(!loadfrmcache)
  {
    global2sm<REAL, halo, halo, 1, halo+1, 0, false, true>
                                        (input, smbuffer_buffer_ptr,
                                          p_x, p_y, global_z,
                                          width_x, width_y, width_z,
                                          // tile_x_with_halo
                                          sm_width_x, ps_x,
                                          cpbase_y, cpend_y,ps_y,
                                          LOCAL_TILE_X,tid_x);
  }
  else
  {
    global2sm<REAL, halo, halo, 1, halo+1, 0, false, false>
                                      (input, smbuffer_buffer_ptr,
                                        p_x, p_y, global_z,
                                        width_x, width_y, width_z,
                                        sm_width_x, ps_x,
                                        tid_y-halo, LOCAL_NOCACHE_Y,ps_y,
                                        LOCAL_TILE_X,tid_x);


    global2sm<REAL, halo, halo, 1, halo+1, 0, false, true>
                                      (input, smbuffer_buffer_ptr,
                                        p_x, p_y, global_z,
                                        width_x, width_y, width_z,
                                        sm_width_x, ps_x,
                                        tid_y-LOCAL_NOCACHE_Y+LOCAL_TILE_Y-1,
                                        LOCAL_TILE_Y+halo, ps_y, 
                                        LOCAL_TILE_X,
                                        tid_x);                         
    // // cached region
    // __syncthreads();
    for(int l_y=0; l_y<LOCAL_ITEM_PER_THREAD; l_y++)
    {
      int local_y=l_y+LOCAL_ITEM_PER_THREAD*tid_y;
      // if(local_y>=LOCAL_TILE_Y-2*LOCAL_NOCACHE_Y) break;
      if(local_y<NOCACHE_Y||local_y>=LOCAL_TILE_Y-NOCACHE_Y)
      {
        continue;
      }
      if(!isloadfrmreg)
      {
        smbuffer_buffer_ptr[halo][(local_y+ps_y)*sm_width_x+ps_x+tid_x]
        // =input[(p_z+halo+cache_z)*width_x*width_y+(local_y+halo+p_y)*width_x+p_x+tid_x];
          = sm_space[frmcachesmid_z*LOCAL_TILE_X*CACHE_TILE_Y+(local_y-NOCACHE_Y)*LOCAL_TILE_X+tid_x];
      }
      else
      {
        smbuffer_buffer_ptr[halo][(local_y+ps_y)*sm_width_x+ps_x+tid_x]
          // =input[(p_z+halo+cache_z)*width_x*width_y+(local_y+halo+p_y)*width_x+p_x+tid_x];
          =  r_space[frmcacheregid_z][l_y];
      }
    }
    // east west
    for(int l_y=threadIdx.x; l_y<CACHE_TILE_Y; l_y+=blockDim.x)
    {
      #pragma unroll
      for(int l_x=0; l_x<halo; l_x++)
      {
        //east
        smbuffer_buffer_ptr[halo][(l_y+LOCAL_NOCACHE_Y+ps_y)*sm_width_x+ps_x+l_x+LOCAL_TILE_X]
          = boundary_buffer[boundary_east_step + (boundary_buffer_index_z) * CACHE_TILE_Y + (l_y) + l_x * boundary_step_yz];
        //west
        smbuffer_buffer_ptr[halo][(l_y+LOCAL_NOCACHE_Y+ps_y)*sm_width_x+ps_x-halo+l_x] 
          = boundary_buffer[boundary_west_step + (boundary_buffer_index_z) * CACHE_TILE_Y + (l_y) + l_x * boundary_step_yz];
      }
    }
    __syncthreads();
  }
      // __syncthreads();
  // __syncthreads();
  //sm2reg

  sm2regs<REAL, LOCAL_ITEM_PER_THREAD, REG_FOLDER_Z, 
            SM_SIZE_Z, halo, 0, halo*2, 
            LOCAL_ITEM_PER_THREAD, 1>
    (smbuffer_buffer_ptr, r_smbuffer, 
      ps_y+index_y, ps_x, 
      // tile_x_with_halo
      sm_width_x, tid_x);
  REAL sum[LOCAL_ITEM_PER_THREAD];
  // #pragma unroll
  _Pragma("unroll")
  for(int l_y=0; l_y<LOCAL_ITEM_PER_THREAD; l_y++)
  {
    sum[l_y]=0;
  }
  //main computation
  computation<REAL,LOCAL_ITEM_PER_THREAD,halo,REG_SIZE_Z>( sum,
                                  smbuffer_buffer_ptr[0],
                                  ps_y+index_y, 
                                  // tile_x_with_halo,
                                  sm_width_x,
                                  tid_x+ps_x,
                                  r_smbuffer,
                                  stencilParaInput);
  // reg 2 ptr
  // out(global, global_z)
  if(!storetocache)
  {
    reg2global3d<REAL, LOCAL_ITEM_PER_THREAD>(
          sum, output,
          global_z, width_z,
          p_y+index_y, width_y,
          p_x, width_x,
          tid_x);
  }  
  else
  {
    _Pragma("unroll")
    for(int l_y=0; l_y<LOCAL_ITEM_PER_THREAD; l_y++)
    {
      int local_y=l_y+index_y;
      // output[global_z*width_x*width_y+(p_y+local_y)*width_x+p_x+tid_x]=sum[l_y];
      if(local_y<LOCAL_NOCACHE_Y||local_y>=LOCAL_TILE_Y-LOCAL_NOCACHE_Y)
      {
        output[global_z*width_x*width_y+(p_y+local_y)*width_x+p_x+tid_x]=sum[l_y];
        continue;
      }
      if(!isstoretoreg)
      {
        sm_space[(tocachesmid_z)*LOCAL_TILE_X*CACHE_TILE_Y+(local_y-LOCAL_NOCACHE_Y)*LOCAL_TILE_X+tid_x]
        //sm_space[(cache_z-halo)*LOCAL_TILE_X*CACHE_TILE_Y+(local_y-LOCAL_NOCACHE_Y)*LOCAL_TILE_X+tid_x]
          =  sum[l_y];
      }
      else
      {
        r_space[tocacheregid_z][l_y]=sum[l_y];
      }

    }
  }
  REAL* tmp = smbuffer_buffer_ptr[0];
  // smswap 
  _Pragma("unroll")
  for(int hl=1; hl<halo+1; hl++)
  {
    smbuffer_buffer_ptr[hl-1]=smbuffer_buffer_ptr[hl];
  }
  smbuffer_buffer_ptr[halo]=tmp;
  regsself3d<REAL,REG_SIZE_Z,LOCAL_ITEM_PER_THREAD>(r_smbuffer);
}