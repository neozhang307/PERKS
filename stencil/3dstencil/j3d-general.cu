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

template<class REAL, int halo, 
int LOCAL_ITEM_PER_THREAD, int LOCAL_TILE_X, int LOCAL_TILE_Y, const int reg_folder_z, bool UseSMCache>
__global__ void 
kernel3d_general(REAL * __restrict__ input, 
                                REAL * __restrict__ output, 
                                int width_z, int width_y, int width_x,
                                REAL* l2_cache_i, REAL* l2_cache_o,
                                int iteration,
                                int max_sm_flder) 
{
  if(!UseSMCache) max_sm_flder=0;
  #define UseRegCache (reg_folder_z!=0)

  const int tile_x_with_halo=LOCAL_TILE_X+2*halo;
  const int tile_y_with_halo=LOCAL_TILE_Y+2*halo;
  stencilParaT;
  
  /****sm related*******/
  extern __shared__ char sm[];
  REAL* sm_rbuffer = (REAL*)sm+1;
 
  REAL* smbuffer_buffer_ptr[halo+1];
  smbuffer_buffer_ptr[0]=sm_rbuffer;
  #pragma unroll
  for(int hl=1; hl<halo+1; hl++)
  {
    smbuffer_buffer_ptr[hl]=smbuffer_buffer_ptr[hl-1]+tile_x_with_halo*tile_y_with_halo;
  }

  REAL* sm_space = sm_rbuffer+tile_x_with_halo*tile_y_with_halo*(halo+1);
  REAL* boundary_buffer=sm_space+max_sm_flder*LOCAL_TILE_X*(CACHE_TILE_Y);
  /**********************/
  /*******register*******/
  register REAL r_smbuffer[2*halo+1][LOCAL_ITEM_PER_THREAD];
  register REAL r_space[reg_folder_z<=0?1:reg_folder_z][LOCAL_TILE_Y-2*halo];
  /***********************/
  const int tid_x = threadIdx.x%LOCAL_TILE_X;
  const int tid_y = threadIdx.x/LOCAL_TILE_X;
  const int dim_y = LOCAL_TILE_Y/LOCAL_ITEM_PER_THREAD;

  const int cpblocksize_y=(tile_y_with_halo)/dim_y;
  const int cpquotion_y=(tile_y_with_halo)%dim_y;

  const int index_y = LOCAL_ITEM_PER_THREAD*tid_y;

  const int cpbase_y = -halo+tid_y*cpblocksize_y+(tid_y<=cpquotion_y?tid_y:cpquotion_y);
  const int cpend_y = cpbase_y + cpblocksize_y + (tid_y<=cpquotion_y?1:0);

  const int ps_y = halo;
  const int ps_x = halo;
  // const int ps_z = halo;

  const int p_x = blockIdx.x * LOCAL_TILE_X;
  const int p_y = blockIdx.y * LOCAL_TILE_Y;

  int blocksize_z=(width_z/gridDim.z);
  int z_quotient = width_z%gridDim.z;

  const int p_z =  blockIdx.z * (blocksize_z) + (blockIdx.z<=z_quotient?blockIdx.z:z_quotient);
  blocksize_z += (blockIdx.z<z_quotient?1:0);
  // const int p_z_reg_start=p_z+halo;
  const int p_z_sm_start=p_z+NOCACHE_Z + reg_folder_z;
  const int p_z_sm_end=p_z_sm_start+max_sm_flder;
  const int p_z_end = p_z + (blocksize_z);
  const int total_folder_z=max_sm_flder+reg_folder_z;

  const int boundary_east_step=0;
  const int boundary_west_step=CACHE_TILE_Y*(total_folder_z);
  const int boundary_step_yz=CACHE_TILE_Y*total_folder_z;
  // #define BD_STEP_XY (BD_TILE_Y*Halo)
// #define BD_STEP_YZ (BD_TILE_Y*FOLDER_Z)
  //global 2 cache
  for(int cache_z=0; cache_z<max_sm_flder; cache_z++)
  {
    for(int l_y=0; l_y<ITEM_PER_THREAD; l_y++)
    {
      int local_y=l_y+ITEM_PER_THREAD*tid_y;
      if(local_y>=CACHE_TILE_Y) break;
      sm_space[cache_z*LOCAL_TILE_X*CACHE_TILE_Y+local_y*LOCAL_TILE_X+tid_x]
        = input[(p_z_sm_start + cache_z)*width_x*width_y+(local_y+NOCACHE_Y+p_y)*width_x+p_x+tid_x];
    }
  }
  // boundary
  //  _Pragma("unroll")
  for(int cache_z=0; cache_z<max_sm_flder+reg_folder_z; cache_z++)
  {
    // #pragma unroll
    // _Pragma("unroll")
    for(int l_y=threadIdx.x; l_y<CACHE_TILE_Y; l_y+=blockDim.x)
    {
      #pragma unroll
      for(int l_x=0; l_x<halo; l_x++)
      {
        //east
        int global_x = p_x+LOCAL_TILE_X+l_x;
        global_x = MIN(width_x-1,global_x);
        boundary_buffer[boundary_east_step + cache_z * CACHE_TILE_Y + (l_y) + l_x * boundary_step_yz]
          = 
            input[(p_z+NOCACHE_Z+cache_z)*width_x*width_y+(p_y+l_y+NOCACHE_Y)*width_x+global_x];
        // //west
        global_x = p_x-halo+l_x;
        global_x = MAX(0,global_x);
        boundary_buffer[boundary_west_step + cache_z * CACHE_TILE_Y + (l_y) + l_x * boundary_step_yz]
          = 
            input[(p_z+NOCACHE_Z+cache_z)*width_x*width_y+(p_y+l_y+NOCACHE_Y)*width_x+global_x];
      }
    }
  }
  // int smz_ind=0;
  cg::grid_group gg = cg::this_grid();
  for(int iter=0; iter<iteration; iter++)
  {
    // halo in(global, 0,halo)
    global2regs3d<REAL, LOCAL_ITEM_PER_THREAD, 1+2*halo>
      (input, r_smbuffer, p_z-halo,width_z, p_y+index_y, width_y, p_x, width_x,tid_x);
    global2sm<REAL, halo, 0, halo, halo+1, 0, true, false>
                                        (input, smbuffer_buffer_ptr,
                                          p_x, p_y, p_z,
                                          width_x, width_y, width_z,
                                          tile_x_with_halo, ps_x,
                                          cpbase_y, cpend_y,ps_y,
                                          LOCAL_TILE_X, tid_x);
    __syncthreads();
    //normal
                                          
    for(int global_z=p_z; global_z<p_z_sm_start-halo; global_z+=1)
    // for(int global_z=p_z; global_z<p_z_end; global_z+=1)
    {
      __syncthreads();
      //in(global, halo+glboal_z)
      global2sm<REAL, halo, halo, 1, halo+1, 0, false, true>
                                        (input, smbuffer_buffer_ptr,
                                          p_x, p_y, global_z,
                                          width_x, width_y, width_z,
                                          tile_x_with_halo, ps_x,
                                          cpbase_y, cpend_y,ps_y,
                                          LOCAL_TILE_X,tid_x);
      __syncthreads();
      //sm2reg
      sm2regs<REAL, LOCAL_ITEM_PER_THREAD, 1+2*halo, 
                1+halo, halo, 0, halo*2, 
                LOCAL_ITEM_PER_THREAD, 1>
        (smbuffer_buffer_ptr, r_smbuffer, 
          ps_y+index_y, ps_x, 
          tile_x_with_halo, tid_x);
      REAL sum[LOCAL_ITEM_PER_THREAD];
      // #pragma unroll
      _Pragma("unroll")
      for(int l_y=0; l_y<LOCAL_ITEM_PER_THREAD; l_y++)
      {
        sum[l_y]=0;
      }
      //main computation
      computation<REAL,LOCAL_ITEM_PER_THREAD,halo>( sum,
                                      smbuffer_buffer_ptr[0],
                                      ps_y+index_y, tile_x_with_halo, tid_x+ps_x,
                                      r_smbuffer,
                                      stencilParaInput);
      // reg 2 ptr
      // out(global, global_z)
      reg2global3d<REAL, LOCAL_ITEM_PER_THREAD>(
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
      regsself3d<REAL,2*halo+1,LOCAL_ITEM_PER_THREAD>(r_smbuffer);
    }
    // sm -> global
    // halo in(sm, halo,halo*2) out(global, 0, halo)
    for(int global_z=p_z_sm_start-halo, cache_z=0; global_z<p_z_sm_start; global_z+=1, cache_z++)
    {
      __syncthreads();
      //in(global, halo+glboal_z)
      // global2sm<REAL, halo, halo, 1, halo+1, 0, false, true>
      //                                   (input, smbuffer_buffer_ptr,
      //                                     p_x, p_y, global_z,
      //                                     width_x, width_y, width_z,
      //                                     tile_x_with_halo, ps_x,
      //                                     cpbase_y, cpend_y,ps_y,
      //                                     LOCAL_TILE_X,tid_x);
      global2sm<REAL, halo, halo, 1, halo+1, 0, false, true>
                                        (input, smbuffer_buffer_ptr,
                                          p_x, p_y, global_z,
                                          width_x, width_y, width_z,
                                          tile_x_with_halo, ps_x,
                                          tid_y-halo, NOCACHE_Y,ps_y,
                                          LOCAL_TILE_X,tid_x);


      global2sm<REAL, halo, halo, 1, halo+1, 0, false, true>
                                        (input, smbuffer_buffer_ptr,
                                          p_x, p_y, global_z,
                                          width_x, width_y, width_z,
                                          tile_x_with_halo, ps_x,
                                          tid_y-NOCACHE_Y+LOCAL_TILE_Y-1,
                                          LOCAL_TILE_Y+halo, ps_y, 
                                          LOCAL_TILE_X,
                                          tid_x);                      
      // __syncthreads();
      // // // cached region
      for(int l_y=0; l_y<ITEM_PER_THREAD; l_y++)
      {
        int local_y=l_y+ITEM_PER_THREAD*tid_y;
        // if(local_y>=CACHE_TILE_Y-1) break;
        if(local_y<NOCACHE_Y||local_y>=TILE_Y-NOCACHE_Y)
        {
          continue;
        }
        smbuffer_buffer_ptr[halo][(local_y+ps_y)*tile_x_with_halo+ps_x+tid_x]
          // =input[(p_z+halo+cache_z)*width_x*width_y+(local_y+halo+p_y)*width_x+p_x+tid_x];
          = sm_space[cache_z*LOCAL_TILE_X*CACHE_TILE_Y+(local_y-NOCACHE_Y)*LOCAL_TILE_X+tid_x];
      }
      // // // east west
      for(int l_y=threadIdx.x; l_y<CACHE_TILE_Y; l_y+=blockDim.x)
      {
        #pragma unroll
        for(int l_x=0; l_x<halo; l_x++)
        {
          int global_x = p_x+LOCAL_TILE_X+l_x;
          global_x = MIN(width_x-1,global_x);
          //east
          smbuffer_buffer_ptr[halo][(l_y+NOCACHE_Y+ps_y)*tile_x_with_halo+ps_x+l_x+LOCAL_TILE_X]
            = boundary_buffer[boundary_east_step + (cache_z+reg_folder_z) * CACHE_TILE_Y + (l_y) + l_x * boundary_step_yz];
            // = input[(p_z+cache_z+(NOCACHE_Z))*width_x*width_y+(p_y+l_y+NOCACHE_Y)*width_x+global_x];
            //west
          smbuffer_buffer_ptr[halo][(l_y+NOCACHE_Y+ps_y)*tile_x_with_halo+ps_x-halo+l_x] 
            = boundary_buffer[boundary_west_step + (cache_z+reg_folder_z) * CACHE_TILE_Y + (l_y) + l_x * boundary_step_yz];
        }
      }
      __syncthreads();
      //sm2reg
      sm2regs<REAL, LOCAL_ITEM_PER_THREAD, 1+2*halo, 1+halo, halo, 0, halo*2, LOCAL_ITEM_PER_THREAD, 1>
                (smbuffer_buffer_ptr, r_smbuffer, ps_y+index_y, ps_x, tile_x_with_halo, tid_x);
      REAL sum[LOCAL_ITEM_PER_THREAD];
      // #pragma unroll
      _Pragma("unroll")
      for(int l_y=0; l_y<LOCAL_ITEM_PER_THREAD; l_y++)
      {
        sum[l_y]=0;
      }
      //main computation
      computation<REAL,LOCAL_ITEM_PER_THREAD,halo>( sum,
                                      smbuffer_buffer_ptr[0],
                                      ps_y+index_y, tile_x_with_halo, tid_x+ps_x,
                                      r_smbuffer,
                                      stencilParaInput);
      // reg 2 ptr
      // out(global, global_z)
      reg2global3d<REAL, LOCAL_ITEM_PER_THREAD>(
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
      regsself3d<REAL,2*halo+1,LOCAL_ITEM_PER_THREAD>(r_smbuffer);
    }
    // // // sm->sm
    // // // in(sm, halo*2, cache_sm_end) out(sm, halo, cache_sm_end-halo)
    for(int global_z=p_z_sm_start, cache_z=halo; global_z<p_z_sm_end-halo; global_z+=1, cache_z++)
    {
      __syncthreads();
      //in(global, halo+glboal_z)
      // global2sm<REAL, halo, halo, 1, halo+1, 0, false, true>
      //                                   (input, smbuffer_buffer_ptr,
      //                                     p_x, p_y, global_z,
      //                                     width_x, width_y, width_z,
      //                                     tile_x_with_halo, ps_x,
      //                                     cpbase_y, cpend_y,ps_y,
      //                                     LOCAL_TILE_X,tid_x);

       global2sm<REAL, halo, halo, 1, halo+1, 0, false, true>
                                        (input, smbuffer_buffer_ptr,
                                          p_x, p_y, global_z,
                                          width_x, width_y, width_z,
                                          tile_x_with_halo, ps_x,
                                          tid_y-halo, NOCACHE_Y,ps_y,
                                          LOCAL_TILE_X,tid_x);


      global2sm<REAL, halo, halo, 1, halo+1, 0, false, true>
                                        (input, smbuffer_buffer_ptr,
                                          p_x, p_y, global_z,
                                          width_x, width_y, width_z,
                                          tile_x_with_halo, ps_x,
                                          tid_y-NOCACHE_Y+LOCAL_TILE_Y-1,
                                          LOCAL_TILE_Y+halo, ps_y, 
                                          LOCAL_TILE_X,
                                          tid_x);                         
      // // cached region
      // __syncthreads();
      for(int l_y=0; l_y<ITEM_PER_THREAD; l_y++)
      {
        int local_y=l_y+ITEM_PER_THREAD*tid_y;
        if(local_y>=CACHE_TILE_Y) break;
        smbuffer_buffer_ptr[halo][(local_y+halo+ps_y)*tile_x_with_halo+ps_x+tid_x]
          // =input[(p_z+halo+cache_z)*width_x*width_y+(local_y+halo+p_y)*width_x+p_x+tid_x];
          = sm_space[cache_z*LOCAL_TILE_X*CACHE_TILE_Y+local_y*LOCAL_TILE_X+tid_x];
      }
      // east west
      for(int l_y=threadIdx.x; l_y<CACHE_TILE_Y; l_y+=blockDim.x)
      {
        #pragma unroll
        for(int l_x=0; l_x<halo; l_x++)
        {
          //east
          smbuffer_buffer_ptr[halo][(l_y+NOCACHE_Y+ps_y)*tile_x_with_halo+ps_x+l_x+LOCAL_TILE_X]
            = boundary_buffer[boundary_east_step + (cache_z+reg_folder_z) * CACHE_TILE_Y + (l_y) + l_x * boundary_step_yz];
          //west
          smbuffer_buffer_ptr[halo][(l_y+NOCACHE_Y+ps_y)*tile_x_with_halo+ps_x-halo+l_x] 
            = boundary_buffer[boundary_west_step + (cache_z+reg_folder_z) * CACHE_TILE_Y + (l_y) + l_x * boundary_step_yz];
        }
      }
       __syncthreads();
      //sm2reg
      sm2regs<REAL, LOCAL_ITEM_PER_THREAD, 1+2*halo, 
                1+halo, halo, 0, halo*2, 
                LOCAL_ITEM_PER_THREAD, 1>
        (smbuffer_buffer_ptr, r_smbuffer, 
          ps_y+index_y, ps_x, 
          tile_x_with_halo, tid_x);
      REAL sum[LOCAL_ITEM_PER_THREAD];
      // #pragma unroll
      _Pragma("unroll")
      for(int l_y=0; l_y<LOCAL_ITEM_PER_THREAD; l_y++)
      {
        sum[l_y]=0;
      }
      // if(threadIdx.x==0&&blockIdx.y==0)
      // //if(threadIdx.x==0&&blockIdx.x==1&&blockIdx.y==0)
      // {
      //   if()
      //   printf("<%d:%f,%f>",global_z,smbuffer_buffer_ptr[0][4*tile_x_with_halo],smbuffer_buffer_ptr[0][4*tile_x_with_halo+halo]);
      // }
      //main computation
      computation<REAL,LOCAL_ITEM_PER_THREAD,halo>( sum,
                                      smbuffer_buffer_ptr[0],
                                      ps_y+index_y, tile_x_with_halo, tid_x+ps_x,
                                      r_smbuffer,
                                      stencilParaInput);
      // reg 2 ptr
      // out(global, global_z)
      // reg2global3d<REAL, LOCAL_ITEM_PER_THREAD>(
      //       sum, output,
      //       global_z, width_z,
      //       p_y+index_y, width_y,
      //       p_x, width_x,
      //       tid_x);

      // _Pragma("unroll")
      // for(int l_y=0; l_y<LOCAL_ITEM_PER_THREAD; l_y++)
      // {
      //   int local_y=l_y+index_y;
      //   output[global_z*width_x*width_y+(p_y+local_y)*width_x+p_x+tid_x]=sum[l_y];
      // }

      _Pragma("unroll")
      for(int l_y=0; l_y<LOCAL_ITEM_PER_THREAD; l_y++)
      {
        int local_y=l_y+ITEM_PER_THREAD*tid_y;
        // output[global_z*width_x*width_y+(p_y+local_y)*width_x+p_x+tid_x]=sum[l_y];
        if(local_y<NOCACHE_Y||local_y>=LOCAL_TILE_Y-NOCACHE_Y)
        {
          output[global_z*width_x*width_y+(p_y+local_y)*width_x+p_x+tid_x]=sum[l_y];
          // output[global_z*width_x*width_y+(p_y+ITEM_PER_THREAD*tid_y+l_y)*width_x+p_x+tid_x]=sum[l_y];
          continue;
        }
        sm_space[(cache_z-halo)*LOCAL_TILE_X*CACHE_TILE_Y+(local_y-NOCACHE_Y)*LOCAL_TILE_X+tid_x]
          = sum[l_y];
      }

      REAL* tmp = smbuffer_buffer_ptr[0];
      // smswap 
      _Pragma("unroll")
      for(int hl=1; hl<halo+1; hl++)
      {
        smbuffer_buffer_ptr[hl-1]=smbuffer_buffer_ptr[hl];
      }
      smbuffer_buffer_ptr[halo]=tmp;
      regsself3d<REAL,2*halo+1,LOCAL_ITEM_PER_THREAD>(r_smbuffer);
    }
    // // // // global->sm
    // // // // in(global, cache_sm_end, cache_sm_end+halo) out(cache_sm_start, cache_sm_end-halo, cache_sm_end)
    
    // // // for(int global_z=p_z_sm_start, cache_z=halo; global_z<p_z_sm_end; global_z+=1, cache_z++)
    for(int global_z=p_z_sm_end-halo, cache_z=max_sm_flder; global_z<p_z_sm_end; global_z+=1, cache_z++)
    // for(int global_z=p_z+3; global_z<p_z_sm_end; global_z+=1)
    {
      __syncthreads();
      //in(global, halo+glboal_z)
      global2sm<REAL, halo, halo, 1, halo+1, 0, false, true>
                                        (input, smbuffer_buffer_ptr,
                                          p_x, p_y, global_z,
                                          width_x, width_y, width_z,
                                          tile_x_with_halo, ps_x,
                                          cpbase_y, cpend_y,ps_y,
                                          LOCAL_TILE_X,tid_x);
      __syncthreads();
      //sm2reg
      sm2regs<REAL, LOCAL_ITEM_PER_THREAD, 1+2*halo, 
                1+halo, halo, 0, halo*2, 
                LOCAL_ITEM_PER_THREAD, 1>
        (smbuffer_buffer_ptr, r_smbuffer, 
          ps_y+index_y, ps_x, 
          tile_x_with_halo, tid_x);
      REAL sum[LOCAL_ITEM_PER_THREAD];
      // #pragma unroll
      _Pragma("unroll")
      for(int l_y=0; l_y<LOCAL_ITEM_PER_THREAD; l_y++)
      {
        sum[l_y]=0;
      }
      //main computation
      computation<REAL,LOCAL_ITEM_PER_THREAD,halo>( sum,
                                      smbuffer_buffer_ptr[0],
                                      ps_y+index_y, tile_x_with_halo, tid_x+ps_x,
                                      r_smbuffer,
                                      stencilParaInput);
      // reg 2 ptr
      // out(global, global_z)
      // reg2global3d<REAL, LOCAL_ITEM_PER_THREAD>(
      //       sum, output,
      //       global_z, width_z,
      //       p_y+index_y, width_y,
      //       p_x, width_x,
      //       tid_x);
      // _Pragma("unroll")
      // for(int l_y=0; l_y<LOCAL_ITEM_PER_THREAD; l_y++)
      // {
      //   int local_y=l_y+index_y;
      //   output[global_z*width_x*width_y+(p_y+local_y)*width_x+p_x+tid_x]=sum[l_y];
      // }

      _Pragma("unroll")
      for(int l_y=0; l_y<LOCAL_ITEM_PER_THREAD; l_y++)
      {
        int local_y=l_y+ITEM_PER_THREAD*tid_y;
        // output[global_z*width_x*width_y+(p_y+local_y)*width_x+p_x+tid_x]=sum[l_y];
        if(local_y<NOCACHE_Y||local_y>=LOCAL_TILE_Y-NOCACHE_Y)
        {
          output[global_z*width_x*width_y+(p_y+local_y)*width_x+p_x+tid_x]=sum[l_y];
          // output[global_z*width_x*width_y+(p_y+ITEM_PER_THREAD*tid_y+l_y)*width_x+p_x+tid_x]=sum[l_y];
          continue;
        }
        sm_space[(cache_z-halo)*LOCAL_TILE_X*CACHE_TILE_Y+(local_y-NOCACHE_Y)*LOCAL_TILE_X+tid_x]
          =  sum[l_y];
      }
      REAL* tmp = smbuffer_buffer_ptr[0];
      // smswap 
      _Pragma("unroll")
      for(int hl=1; hl<halo+1; hl++)
      {
        smbuffer_buffer_ptr[hl-1]=smbuffer_buffer_ptr[hl];
      }
      smbuffer_buffer_ptr[halo]=tmp;
      regsself3d<REAL,2*halo+1,LOCAL_ITEM_PER_THREAD>(r_smbuffer);
    }
    // general version
    // in(global, cache_sm_end+halo, p_z_end+halo) out(global, cache_sm_end, p_z_end) 
    for(int global_z=p_z_sm_end; global_z<p_z_end; global_z+=1)
    // for(int global_z=p_z_sm_start; global_z<p_z_end; global_z+=1)
    {
      __syncthreads();
      //in(global, halo+glboal_z)
      global2sm<REAL, halo, halo, 1, halo+1, 0, false, true>
                                        (input, smbuffer_buffer_ptr,
                                          p_x, p_y, global_z,
                                          width_x, width_y, width_z,
                                          tile_x_with_halo, ps_x,
                                          cpbase_y, cpend_y,ps_y,
                                          LOCAL_TILE_X,tid_x);
      __syncthreads();
      //sm2reg
      sm2regs<REAL, LOCAL_ITEM_PER_THREAD, 1+2*halo, 
                1+halo, halo, 0, halo*2, 
                LOCAL_ITEM_PER_THREAD, 1>
        (smbuffer_buffer_ptr, r_smbuffer, 
          ps_y+index_y, ps_x, 
          tile_x_with_halo, tid_x);
      REAL sum[LOCAL_ITEM_PER_THREAD];
      // #pragma unroll
      _Pragma("unroll")
      for(int l_y=0; l_y<LOCAL_ITEM_PER_THREAD; l_y++)
      {
        sum[l_y]=0;
      }
      //main computation
      computation<REAL,LOCAL_ITEM_PER_THREAD,halo>( sum,
                                      smbuffer_buffer_ptr[0],
                                      ps_y+index_y, tile_x_with_halo, tid_x+ps_x,
                                      r_smbuffer,
                                      stencilParaInput);
      // reg 2 ptr
      // out(global, global_z)
      reg2global3d<REAL, LOCAL_ITEM_PER_THREAD>(
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
      regsself3d<REAL,2*halo+1,LOCAL_ITEM_PER_THREAD>(r_smbuffer);
    }
   
    //sm
    for(int l_z=0; l_z<max_sm_flder; l_z++)
    {
      for(int l_x=0; l_x<halo; l_x++)
      {
        for(int l_y=threadIdx.x; l_y<CACHE_TILE_Y;l_y+=blockDim.x)
        {
          //east
          boundary_buffer[boundary_east_step + (l_z+reg_folder_z) * CACHE_TILE_Y + (l_y) + l_x * boundary_step_yz]
          =
          sm_space[l_z*(CACHE_TILE_Y)*LOCAL_TILE_X+(l_y)*LOCAL_TILE_X+LOCAL_TILE_X-halo+l_x];

          //west
          boundary_buffer[boundary_west_step + (l_z+reg_folder_z) * CACHE_TILE_Y + (l_y) + l_x * boundary_step_yz]
          =
          sm_space[l_z*(CACHE_TILE_Y)*LOCAL_TILE_X+(l_y)*LOCAL_TILE_X+l_x];
        }
      }
    }
    __syncthreads();

      

    if(iter>=iteration-1)break;
    //deal with east and west boundary
    //store to global memory in l2 cache pointer (hopefully)  
   
    {
      // int blocksize_z=(width_z/gridDim.z);
      // int z_quotient = width_z%gridDim.z;

      // const int p_z =  blockIdx.z * (blocksize_z) + (blockIdx.z<=z_quotient?blockIdx.z:z_quotient);

      int bid_x=blockIdx.x;
      int gdimx=gridDim.x;

      int bid_y=blockIdx.y;
      int gdimy=gridDim.y;
      
      int bid_z=blockIdx.z;
      // int p_z = blockIdx.z*total_folder_z;
      //x
      // #pragma unroll
      // _Pragma("unroll")
      for(int l_x=0; l_x<halo; l_x++)
      {
        //z
        // #pragma unroll
        // _Pragma("unroll")
        for(int l_z=0; l_z<total_folder_z; l_z++)
        {
          //y
          // #pragma unroll
          // _Pragma("unroll")
          for(int l_y=threadIdx.x; l_y<CACHE_TILE_Y; l_y+=blockDim.x)
          {
            // //west
            // l2_cache_o[((bid_x*2+0)*halo+l_x)*width_y*width_z + (l_z+p_z+1) * CACHE_TILE_Y*dim_y + (l_y+bid_y*LOCAL_TILE_Y)]
            l2_cache_o[ ((bid_x+bid_y*gdimx+ bid_z*gdimx*gdimy)*2+0)*halo*total_folder_z*CACHE_TILE_Y + (l_z) * CACHE_TILE_Y + (l_y) + l_x * boundary_step_yz] 
            = boundary_buffer[boundary_west_step + (l_z) * CACHE_TILE_Y + (l_y) + l_x * boundary_step_yz];
            //east
            // l2_cache_o[((bid_x*2+1)*halo+l_x)*width_y*width_z + (l_z+p_z+1) * CACHE_TILE_Y*dim_y + (l_y+bid_y*LOCAL_TILE_Y)]
            l2_cache_o[ ((bid_x+bid_y*gdimx+ bid_z*gdimx*gdimy)*2+1)*halo*total_folder_z*CACHE_TILE_Y + (l_z) * CACHE_TILE_Y + (l_y) + l_x * boundary_step_yz]  
            = boundary_buffer[boundary_east_step + (l_z) * CACHE_TILE_Y + (l_y) + l_x * boundary_step_yz];
          }
        }
      }
    }

    gg.sync();
    //load frm global memory in l2 cache pointer (hopefully)
    REAL* tmp_ptr =output;
    output=input;
    input=tmp_ptr;

    tmp_ptr=l2_cache_o;
    l2_cache_o=l2_cache_i;
    l2_cache_i=tmp_ptr;

    {
      // int blocksize_z=(width_z/gridDim.z);
      // int z_quotient = width_z%gridDim.z;

      // const int p_z =  blockIdx.z * (blocksize_z) + (blockIdx.z<=z_quotient?blockIdx.z:z_quotient);

      // int bid_y=blockIdx.x/gdim_x;
      int bid_x=blockIdx.x;
      int gdimx=gridDim.x;

      int bid_y=blockIdx.y;
      int gdimy=gridDim.y;
      
      int bid_z=blockIdx.z;
      //x
      // #pragma unroll
      // _Pragma("unroll")
      for(int l_x=0; l_x<halo; l_x++)
      {
        //z
        // #pragma unroll
        // _Pragma("unroll")
        for(int l_z=0; l_z<total_folder_z; l_z++)
        {
          //y
          // #pragma unroll
          // _Pragma("unroll")
          for(int l_y=threadIdx.x; l_y<CACHE_TILE_Y; l_y+=blockDim.x)
          {
            //east
            boundary_buffer[boundary_east_step + (l_z) * CACHE_TILE_Y + (l_y) + l_x * boundary_step_yz] =
              (bid_x==gdimx-1?
             boundary_buffer[boundary_east_step + (l_z) * CACHE_TILE_Y + (l_y) + (halo-1) * boundary_step_yz]
             // l2_cache_i[((bid_x*2+1)*Halo+Halo-1)*width_y*width_z + (l_z+p_z+1) * TILE_Y*gdim_y + (l_y+bid_y*TILE_Y)]
             :
             l2_cache_i[((bid_x+1+bid_y*gdimx+ bid_z*gdimx*gdimy)*2+0)*halo*total_folder_z*CACHE_TILE_Y + (l_z) * CACHE_TILE_Y + (l_y) + l_x * boundary_step_yz]
             //l2_cache_i[(((bid_x+1)*2+0)*Halo+l_x)   *width_y*width_z + (l_z+p_z+1) * TILE_Y*gdim_y + (l_y+bid_y*TILE_Y)]
             )
             ;
            //l2_cache_o[ ((bid_x+bid_y*gdimx+ bid_z*gdimx*gdimy)*2+1)*halo*total_folder_z*CACHE_TILE_Y + (l_z) * CACHE_TILE_Y + (l_y) + l_x * boundary_step_yz] 
            //  l2_cache_o[ (bid_x+bid_y*gdimx+ bid_z*gdimx*gdimy+1)*halo*total_folder_z*CACHE_TILE_Y*2 + (l_z) * CACHE_TILE_Y + (l_y) + l_x * boundary_step_yz] 
            //west
            boundary_buffer[boundary_west_step + (l_z) * CACHE_TILE_Y + (l_y) + l_x * boundary_step_yz] =
              (bid_x==0?
                boundary_buffer[boundary_west_step + (l_z) * CACHE_TILE_Y + (l_y) + 0 * boundary_step_yz]
                // l2_cache_i[((bid_x*2+0)*Halo+(0))*width_y*width_z + (l_z+p_z+1) * TILE_Y*gdim_y + (l_y+bid_y*TILE_Y)]
                :
                l2_cache_i[((bid_x-1+bid_y*gdimx+ bid_z*gdimx*gdimy)*2+1)*halo*total_folder_z*CACHE_TILE_Y + (l_z) * CACHE_TILE_Y + (l_y) + l_x * boundary_step_yz])
                ;
                // l2_cache_i[(((bid_x-1)*2+1)*Halo+l_x)*width_y*width_z + (l_z+p_z+1) * TILE_Y*gdim_y + (l_y+bid_y*TILE_Y)]);
                //l2_cache_i[(((bid_x-1)*2+1)*Halo+l_x)*width_y*width_z + (l_z+p_z+1) * TILE_Y*gdim_y + (l_y+bid_y*TILE_Y)]);
          }
        }
      }
      __syncthreads();
    }
  }

   
  // for(int cache_z=0; cache_z<max_sm_flder; cache_z++)
  for(int global_z=p_z_sm_start, cache_z=halo; global_z<p_z_sm_end; global_z+=1, cache_z++)
  {

      // _Pragma("unroll")
      // for(int l_y=0; l_y<LOCAL_ITEM_PER_THREAD; l_y++)
      // {
      //   int local_y=l_y+ITEM_PER_THREAD*tid_y;
      //   if(local_y<NOCACHE_Y&&local_y>=LOCAL_TILE_Y-NOCACHE_Y)
      //   {
      //     continue;
      //   }
      //   output[global_z*width_x*width_y+(p_y+local_y)*width_x+p_x+tid_x]=
      //     sm_space[(cache_z-halo)*LOCAL_TILE_X*CACHE_TILE_Y+(local_y-NOCACHE_Y)*LOCAL_TILE_X+tid_x];

      // }   
      _Pragma("unroll")
      for(int l_y=0; l_y<LOCAL_ITEM_PER_THREAD; l_y++)
      {
        int local_y=l_y+ITEM_PER_THREAD*tid_y;
        // output[global_z*width_x*width_y+(p_y+local_y)*width_x+p_x+tid_x]=sum[l_y];
        if(local_y<NOCACHE_Y||local_y>=LOCAL_TILE_Y-NOCACHE_Y)
        {
          // output[global_z*width_x*width_y+(p_y+local_y)*width_x+p_x+tid_x]=sum[l_y];
          // output[global_z*width_x*width_y+(p_y+ITEM_PER_THREAD*tid_y+l_y)*width_x+p_x+tid_x]=sum[l_y];
          continue;
        }
        output[global_z*width_x*width_y+(p_y+local_y)*width_x+p_x+tid_x]  =
          sm_space[(cache_z-halo)*LOCAL_TILE_X*CACHE_TILE_Y+(local_y-NOCACHE_Y)*LOCAL_TILE_X+tid_x];
      }  
      // break; 
  }
 
  #undef UseRegCache
}


template __global__ void kernel3d_general<float,HALO,ITEM_PER_THREAD,TILE_X,TILE_Y> 
    (float *__restrict__, float *__restrict__ , int , int , int, float*,float*,int,int);

template __global__ void kernel3d_general<float,HALO,ITEM_PER_THREAD,TILE_X,TILE_Y,0,true> 
    (float *__restrict__, float *__restrict__ , int , int , int, float*,float*,int,int);

template __global__ void kernel3d_general<float,HALO,ITEM_PER_THREAD,TILE_X,TILE_Y,REG_FOLDER_Z,true> 
    (float *__restrict__, float *__restrict__ , int , int , int, float*,float*,int,int);
    // template __global__ void kernel3d_general<double,HALO,ITEM_PER_THREAD,TILE_X,TILE_Y> 
//     (double *__restrict__, double *__restrict__ , int , int , int , double*,double*,int,int);

// #ifndef PERSISTENT 
//   PERKS_INITIALIZE_ALL_TYPE_1ARG(PERKS_DECLARE_INITIONIZATION_BASELINE,HALO);
// #else
//   PERKS_INITIALIZE_ALL_TYPE_1ARG(PERKS_DECLARE_INITIONIZATION_PERSISTENT,HALO);
// #endif