// #include "./config.cuh"
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

template<class REAL, int halo, int LOCAL_ITEM_PER_THREAD, int LOCAL_TILE_X, int LOCAL_TILE_Y, int SM_SIZE_Z, int REG_SIZE_Z,
          int REG_CACHESIZE_Z=1,          
          bool loadfrmcache=false, bool storetocache=false,
          bool isloadfrmreg=false, bool isstoretoreg=false,
          int REGY_SIZE=REG_Y_SIZE_MOD, int REGX_SIZE=2*halo+1>
__device__ void __forceinline__ process_one_layer 
      (
        REAL * __restrict__ input, REAL * __restrict__ output, 
        REAL* smbuffer_buffer_ptr[SM_SIZE_Z],

        // REAL r_smbuffer[REG_SIZE_Z][LOCAL_ITEM_PER_THREAD],
#ifndef BOX
        REAL r_smbuffer[REG_SIZE_Z][LOCAL_ITEM_PER_THREAD],
#else
        REAL r_smbuffer[REG_SIZE_Z][REGY_SIZE][REGX_SIZE],
#endif   
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
        const int boundary_north_step=0, const int boundary_south_step=0,
        const int boundary_step_yz=0
        
      )
{
  // __syncthreads();
  //in(global, halo+glboal_z)
  if(!loadfrmcache)
  {
    // global2sm<REAL, halo, halo, 1, halo+1, 0, false, true>
    global2sm<REAL, halo, halo, 1, halo+1+isBOX, halo+isBOX, false, true>
                                        (input, smbuffer_buffer_ptr,
                                          p_x, p_y, global_z,
                                          width_x, width_y, width_z,
                                          // tile_x_with_halo
                                          sm_width_x, ps_x,
                                          cpbase_y, cpend_y,1,ps_y,
                                          LOCAL_TILE_X,tid_x);
      //sm2reg
  }
  else
  {
                     
    // // cached region
    for(int l_y=0; l_y<LOCAL_ITEM_PER_THREAD; l_y++)
    {
      int local_y=l_y+LOCAL_ITEM_PER_THREAD*tid_y;

      if(!isloadfrmreg)
      {
        smbuffer_buffer_ptr[halo+isBOX][(local_y+ps_y)*sm_width_x+ps_x+tid_x]
        // =input[(p_z+halo+cache_z)*width_x*width_y+(local_y+halo+p_y)*width_x+p_x+tid_x];
          = sm_space[frmcachesmid_z*LOCAL_TILE_X*LOCAL_TILE_Y+(local_y-0)*LOCAL_TILE_X+tid_x];
      }
      else
      {
        smbuffer_buffer_ptr[halo+isBOX][(local_y+ps_y)*sm_width_x+ps_x+tid_x]
          // =input[(p_z+halo+cache_z)*width_x*width_y+(local_y+halo+p_y)*width_x+p_x+tid_x];
          =  r_space[frmcacheregid_z][l_y];
      }
    }
      //sm2reg
    // east west
    for(int l_y=tid_x-isBOX; l_y<LOCAL_TILE_Y+isBOX; l_y+=LOCAL_TILE_X)
    {
      #pragma unroll
      for(int l_x=tid_y; l_x<halo; l_x+=LOCAL_TILE_Y/LOCAL_ITEM_PER_THREAD)
      {
        //east
        smbuffer_buffer_ptr[halo+isBOX][(l_y+0+ps_y)*sm_width_x+ps_x+l_x+LOCAL_TILE_X]
          = boundary_buffer[boundary_east_step + (boundary_buffer_index_z) * (LOCAL_TILE_Y+2*isBOX)*halo  + (l_y+isBOX) + l_x * (LOCAL_TILE_Y+2*isBOX)];
        //west
        smbuffer_buffer_ptr[halo+isBOX][(l_y+0+ps_y)*sm_width_x+ps_x-halo+l_x] 
          = boundary_buffer[boundary_west_step + (boundary_buffer_index_z) * (LOCAL_TILE_Y+2*isBOX)*halo  + (l_y+isBOX) + l_x * (LOCAL_TILE_Y+2*isBOX)];
      }
    }
    //north south
    {
      int l_x=tid_x;
      #pragma unroll
      for(int l_y=tid_y; l_y<halo; l_y+=bdimx/LOCAL_TILE_X)
      {
        //north
        smbuffer_buffer_ptr[halo+isBOX][(l_y+LOCAL_TILE_Y+ps_y)*sm_width_x+ps_x+l_x]
          = boundary_buffer[boundary_north_step + (boundary_buffer_index_z) * (LOCAL_TILE_X)*halo  + (l_x) + l_y * (LOCAL_TILE_X)];
        
        //south
        //->
        smbuffer_buffer_ptr[halo+isBOX][(l_y-halo+ps_y)*sm_width_x+ps_x+l_x] 
          = boundary_buffer[boundary_south_step + (boundary_buffer_index_z) * (LOCAL_TILE_X)*halo  + (l_x) + l_y * (LOCAL_TILE_X)];
      }
    }
    __syncthreads();
  }

  //sm2reg
  #ifndef BOX
  sm2regs<REAL, LOCAL_ITEM_PER_THREAD, 1+2*halo, 
                  1+halo, halo, 
                  0, halo*2, 
                  LOCAL_ITEM_PER_THREAD, 1>
    (smbuffer_buffer_ptr, r_smbuffer, 
      ps_y+index_y, ps_x, 
      sm_width_x, tid_x);

  #else
  #endif
  REAL sum[LOCAL_ITEM_PER_THREAD];
  // #pragma unroll
  _Pragma("unroll")
  for(int l_y=0; l_y<LOCAL_ITEM_PER_THREAD; l_y++)
  {
    sum[l_y]=0;
  }
  //main computation
  computation<REAL,LOCAL_ITEM_PER_THREAD,halo>( sum,
                                  smbuffer_buffer_ptr,
                                  ps_y+index_y, 
                                  sm_width_x,
                                  tid_x+ps_x,
                                  r_smbuffer,
                                  stencilParaInput);

  // #ifdef BOX
  __syncthreads();
  // #endif
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

      // if(!isstoretoreg)
      if(!isstoretoreg)
      {
        sm_space[(tocachesmid_z)*LOCAL_TILE_X*LOCAL_TILE_Y+(local_y-0)*LOCAL_TILE_X+tid_x]
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
  for(int hl=1; hl<halo+1+isBOX; hl++)
  {
    smbuffer_buffer_ptr[hl-1]=smbuffer_buffer_ptr[hl];
  }
  smbuffer_buffer_ptr[halo+isBOX]=tmp;
  #ifndef BOX
    regsself3d<REAL,REG_SIZE_Z,LOCAL_ITEM_PER_THREAD>(r_smbuffer);
  #else
  #endif
}