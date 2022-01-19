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

// #define NOCACHE_Y (0)
#define NOCACHE_Z (HALO)
// #define LOCAL_TILE_Y (TILE_Y-2*NOCACHE_Y)
#include "./j3d-general-kernels.cuh"

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
 
  REAL* smbuffer_buffer_ptr[halo+1+isBOX];
  smbuffer_buffer_ptr[0]=sm_rbuffer;
  #pragma unroll
  for(int hl=1; hl<halo+1+isBOX; hl++)
  {
    smbuffer_buffer_ptr[hl]=smbuffer_buffer_ptr[hl-1]+tile_x_with_halo*tile_y_with_halo;
  }

  REAL* sm_space = sm_rbuffer+tile_x_with_halo*tile_y_with_halo*(halo+1+isBOX);
  REAL* boundary_buffer=sm_space+max_sm_flder*LOCAL_TILE_X*(LOCAL_TILE_Y);
  /**********************/
  /*******register*******/
  // register REAL r_smbuffer[2*halo+1][LOCAL_ITEM_PER_THREAD];
  #ifndef BOX
    register REAL r_smbuffer[2*halo+1][REG_Y_SIZE_MOD];
  #else 
    register REAL r_smbuffer[2*halo+1][REG_Y_SIZE_MOD][2*halo+1];
  #endif
  register REAL r_space[reg_folder_z<=0?1:reg_folder_z][LOCAL_ITEM_PER_THREAD];
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
  const int p_z_reg_start=p_z+NOCACHE_Z;
  const int p_z_sm_start=p_z+NOCACHE_Z + reg_folder_z;
  const int p_z_sm_end=p_z_sm_start+max_sm_flder;
  const int p_z_end = p_z + (blocksize_z);
  const int total_folder_z=max_sm_flder+reg_folder_z;

  const int boundary_east_step=0;
  const int boundary_west_step=(LOCAL_TILE_Y+isBOX*2)*(total_folder_z)*halo;
  const int boundary_north_step=boundary_west_step+(LOCAL_TILE_Y+isBOX*2)*(total_folder_z)*halo;
  const int boundary_south_step=boundary_north_step+TILE_X*total_folder_z*halo;

  const int l2_boundary_east_step=0;
  const int l2_boundary_west_step=width_y*halo*gridDim.x*total_folder_z*gridDim.z;
  const int l2_boundary_north_step=l2_boundary_west_step+width_y*halo*gridDim.x*total_folder_z*gridDim.z;
  const int l2_boundary_south_step=l2_boundary_north_step+width_x*halo*gridDim.y*total_folder_z*gridDim.z; 

  const int boundary_step_yz=(LOCAL_TILE_Y+isBOX*2)*total_folder_z;
  // const int boundary_step_xz=halo*total_folder_z;
  
  // #define BD_STEP_XY (BD_TILE_Y*Halo)
// #define BD_STEP_YZ (BD_TILE_Y*FOLDER_Z)
  // global 2 reg cache  
  _Pragma("unroll")
  for(int cache_z=0; cache_z<reg_folder_z; cache_z++)
  {
    for(int l_y=0; l_y<LOCAL_ITEM_PER_THREAD; l_y++)
    {
      int local_y=l_y+LOCAL_ITEM_PER_THREAD*tid_y;
      r_space[cache_z][l_y]
        = input[(p_z_reg_start + cache_z)*width_x*width_y+(local_y+p_y)*width_x+p_x+tid_x];
    }
  }
  // global 2 sm cache
  for(int cache_z=0; cache_z<max_sm_flder; cache_z++)
  {
    for(int l_y=0; l_y<LOCAL_ITEM_PER_THREAD; l_y++)
    {
      int local_y=l_y+LOCAL_ITEM_PER_THREAD*tid_y;
      sm_space[cache_z*LOCAL_TILE_X*LOCAL_TILE_Y+local_y*LOCAL_TILE_X+tid_x]
        = input[(p_z_sm_start + cache_z)*width_x*width_y+(local_y+p_y)*width_x+p_x+tid_x];
    }
  }
  // boundary
  for(int cache_z=0; cache_z<max_sm_flder+reg_folder_z; cache_z++)
  {
    for(int l_y=threadIdx.x-isBOX; l_y<LOCAL_TILE_Y+isBOX; l_y+=blockDim.x)
    {
      int global_y=MIN(p_y+l_y,width_y-1);
      global_y=MAX(0,global_y);
      #pragma unroll
      for(int l_x=0; l_x<halo; l_x++)
      {
        //east
        int global_x = p_x+LOCAL_TILE_X+l_x;
        global_x = MIN(width_x-1,global_x);
        boundary_buffer[boundary_east_step + cache_z *  (LOCAL_TILE_Y+2*isBOX)*halo + (l_y+isBOX) + l_x * (LOCAL_TILE_Y+2*isBOX)]
          = 
            input[(p_z+NOCACHE_Z+cache_z)*width_x*width_y+(global_y)*width_x+global_x];
        // //west
        global_x = p_x-halo+l_x;
        global_x = MAX(0,global_x);
        boundary_buffer[boundary_west_step + cache_z *  (LOCAL_TILE_Y+2*isBOX)*halo + (l_y+isBOX) + l_x *  (LOCAL_TILE_Y+2*isBOX)]
          = 
            input[(p_z+NOCACHE_Z+cache_z)*width_x*width_y+(global_y)*width_x+global_x];
      }
    }
    for(int l_x=threadIdx.x; l_x<TILE_X; l_x+=blockDim.x)
    {
      int global_x = p_x+l_x;
      #pragma unroll
      for(int l_y=0; l_y<halo; l_y++)
      {
        //north
        int global_y = p_y + LOCAL_TILE_Y + l_y;
        global_y=MIN(global_y,width_y-1);

        boundary_buffer[boundary_north_step + cache_z * LOCAL_TILE_X * halo + (l_x) + l_y * LOCAL_TILE_X]
          = 
            input[(p_z+NOCACHE_Z+cache_z)*width_x*width_y+(global_y)*width_x+global_x];
        
        //south
        global_y = p_y - halo + l_y;
        global_y=MAX(global_y,0);

        boundary_buffer[boundary_south_step + cache_z * LOCAL_TILE_X * halo + (l_x) + l_y * LOCAL_TILE_X]
          = 
            input[(p_z+NOCACHE_Z+cache_z)*width_x*width_y+(global_y)*width_x+global_x];
      }
    }
  }
  // int smz_ind=0;
  cg::grid_group gg = cg::this_grid();
  for(int iter=0; iter<iteration; iter++)
  {
    // halo in(global, 0,halo)
    #ifndef BOX
    global2regs3d<REAL, 1+2*halo, LOCAL_ITEM_PER_THREAD>
      (input, r_smbuffer, p_z-halo,width_z, p_y+index_y, width_y, p_x, width_x,tid_x);
    #else
    #endif
    // global2sm<REAL, halo, 0, halo, halo+1, 0, true, false>
    global2sm<REAL, halo, -isBOX, halo + isBOX, halo+isBOX+1, 0, true, false>
                                        (input, smbuffer_buffer_ptr,
                                          p_x, p_y, p_z,
                                          width_x, width_y, width_z,
                                          tile_x_with_halo, ps_x,
                                          cpbase_y, cpend_y, 1,ps_y,
                                          LOCAL_TILE_X, tid_x);
    // // __syncthreads();
    // //normal
    // // // reg->global
    if(UseRegCache)
    {
      _Pragma("unroll")
      for(int global_z=p_z, cache_z_reg=0; global_z<p_z_reg_start; global_z+=1, cache_z_reg++)
      {
        process_one_layer<REAL, halo, LOCAL_ITEM_PER_THREAD, LOCAL_TILE_X, LOCAL_TILE_Y, halo+1+isBOX, 2*halo+1,
          reg_folder_z==0?1:reg_folder_z,
          true,false,
          true, false>
        (
          input, output, 
          smbuffer_buffer_ptr, 
          r_smbuffer,

          global_z,  p_y,  p_x,
          width_z,  width_y,  width_x,

          ps_y, ps_x, tile_x_with_halo,
          cpbase_y, cpend_y, index_y,
          tid_x, tid_y,
          stencilParaInput,
          sm_space, 0, 0,
          r_space, cache_z_reg, 0,
          boundary_buffer, cache_z_reg, 
          boundary_east_step, boundary_west_step,
          boundary_north_step, boundary_south_step,
          boundary_step_yz
        );
      }    
      // // reg->reg                    
      _Pragma("unroll")   
      for(int global_z=p_z_reg_start,cache_z_reg=halo; global_z<p_z_sm_start-halo; global_z+=1,cache_z_reg++)
      {
        process_one_layer<REAL, halo, LOCAL_ITEM_PER_THREAD, LOCAL_TILE_X, LOCAL_TILE_Y, halo+1+isBOX, 2*halo+1,
          reg_folder_z==0?1:reg_folder_z,
          true,true,
          true, true>
        (
          input, output, 
          smbuffer_buffer_ptr, 
          r_smbuffer,

          global_z,  p_y,  p_x,
          width_z,  width_y,  width_x,

          ps_y, ps_x, tile_x_with_halo,
          cpbase_y, cpend_y, index_y,
          tid_x, tid_y,
          stencilParaInput,
          sm_space, 0, 0,
          r_space, cache_z_reg, cache_z_reg-halo,
          boundary_buffer, cache_z_reg, 
          boundary_east_step, boundary_west_step,
          boundary_north_step, boundary_south_step,
          boundary_step_yz
        );
      }
    }
    if((UseRegCache)&&UseSMCache)
    {
       // // sm -> (reg)
      // // halo in(sm, halo,halo*2) out(global, 0, halo)
      _Pragma("unroll")
      for(int global_z=p_z_sm_start-halo, cache_z=0; global_z<p_z_sm_start; global_z+=1, cache_z++)
      // for(int global_z=p_z; global_z<p_z_sm_start; global_z+=1)
      {
        process_one_layer<REAL, halo, LOCAL_ITEM_PER_THREAD, LOCAL_TILE_X, LOCAL_TILE_Y, halo+1+isBOX, 2*halo+1,
          reg_folder_z==0?1:reg_folder_z,
          true,true,
          false, true>
        (
          input, output, 
          smbuffer_buffer_ptr, 
          r_smbuffer,

          global_z,  p_y,  p_x,
          width_z,  width_y,  width_x,

          ps_y, ps_x, tile_x_with_halo,
          cpbase_y, cpend_y, index_y,
          tid_x, tid_y,
          stencilParaInput,
          sm_space, cache_z, 0,
          r_space, 0, cache_z+reg_folder_z-halo,
          boundary_buffer, cache_z+reg_folder_z, 
          boundary_east_step, boundary_west_step,
          boundary_north_step, boundary_south_step,
          boundary_step_yz
        );
      }
    }
    
    if((UseRegCache)&&!UseSMCache)
    {
       // // global -> (reg)
      // // halo in(sm, halo,halo*2) out(global, 0, halo)
      _Pragma("unroll")
      for(int global_z=p_z_sm_start-halo, cache_z=0, cache_z_reg=reg_folder_z; global_z<p_z_sm_start; global_z+=1, cache_z++,cache_z_reg++)
      // for(int global_z=p_z; global_z<p_z_sm_start; global_z+=1)
      {
        process_one_layer<REAL, halo, LOCAL_ITEM_PER_THREAD, LOCAL_TILE_X, LOCAL_TILE_Y, halo+1+isBOX, 2*halo+1,
          reg_folder_z==0?1:reg_folder_z,
          false,true,
          false, true>
        (
          input, output, 
          smbuffer_buffer_ptr, 
          r_smbuffer,

          global_z,  p_y,  p_x,
          width_z,  width_y,  width_x,

          ps_y, ps_x, tile_x_with_halo,
          cpbase_y, cpend_y, index_y,
          tid_x, tid_y,
          stencilParaInput,
          sm_space, 0, 0,
          r_space, 0, cache_z_reg-halo,
          boundary_buffer, cache_z+reg_folder_z, 
          boundary_east_step, boundary_west_step,
          boundary_north_step, boundary_south_step,
          boundary_step_yz
        );
      }
    }
    if(!UseRegCache&&UseSMCache)
    {
      // sm -> global
      _Pragma("unroll")
      for(int global_z=p_z_sm_start-halo, cache_z=0, cache_z_reg=reg_folder_z; global_z<p_z_sm_start; global_z+=1, cache_z++,cache_z_reg++)
      // for(int global_z=p_z; global_z<p_z_sm_start; global_z+=1)
      {
        process_one_layer<REAL, halo, LOCAL_ITEM_PER_THREAD, LOCAL_TILE_X, LOCAL_TILE_Y, halo+1+isBOX, 2*halo+1,
          reg_folder_z==0?1:reg_folder_z,
          true, false,
          false, false>(
          input, output, 
          smbuffer_buffer_ptr, 
          r_smbuffer,

          global_z,  p_y,  p_x,
          width_z,  width_y,  width_x,

          ps_y, ps_x, tile_x_with_halo,
          cpbase_y, cpend_y, index_y,
          tid_x, tid_y,
          stencilParaInput,
          sm_space, cache_z, 0,
          r_space, 0, cache_z_reg-halo,
          boundary_buffer, cache_z+reg_folder_z, 
          boundary_east_step, boundary_west_step,
          boundary_north_step, boundary_south_step,
            boundary_step_yz
          );
      }
    }

    if(UseSMCache)
    {
      // // // // sm->sm
      // // // in(sm, halo*2, cache_sm_end) out(sm, halo, cache_sm_end-halo)
      for(int global_z=p_z_sm_start, cache_z=halo; global_z<p_z_sm_end-halo; global_z+=1, cache_z++)
      // for(int global_z=p_z, cache_z=halo; global_z<p_z_sm_end-halo; global_z+=1, cache_z++)
      {
        process_one_layer<REAL, halo, LOCAL_ITEM_PER_THREAD, LOCAL_TILE_X, LOCAL_TILE_Y, halo+1+isBOX, 2*halo+1,
          reg_folder_z==0?1:reg_folder_z,
          true,true>
        (
          input, output, 
          smbuffer_buffer_ptr, 
          r_smbuffer,

          global_z,  p_y,  p_x,
          width_z,  width_y,  width_x,

          ps_y, ps_x, tile_x_with_halo,
          cpbase_y, cpend_y, index_y,
          tid_x, tid_y,
          stencilParaInput,
          sm_space, cache_z, cache_z-halo,
          r_space, 0, 0,
          boundary_buffer, cache_z+reg_folder_z, 
          boundary_east_step, boundary_west_step,
          boundary_north_step, boundary_south_step,
          boundary_step_yz
        );
      }
      // // // // // global->sm
      // // // // // in(global, cache_sm_end, cache_sm_end+halo) out(cache_sm_start, cache_sm_end-halo, cache_sm_end)
      
      // // // for(int global_z=p_z_sm_start, cache_z=halo; global_z<p_z_sm_end; global_z+=1, cache_z++)
      for(int global_z=p_z_sm_end-halo, cache_z=max_sm_flder; global_z<p_z_sm_end; global_z+=1, cache_z++)
      // for(int global_z=p_z; global_z<p_z_sm_end; global_z+=1)
      {
        process_one_layer<REAL, halo, LOCAL_ITEM_PER_THREAD, LOCAL_TILE_X, LOCAL_TILE_Y, halo+1+isBOX, 2*halo+1,
          reg_folder_z==0?1:reg_folder_z,
          false,true>
        (
          input, output, 
          smbuffer_buffer_ptr, 
          r_smbuffer,

          global_z,  p_y,  p_x,
          width_z,  width_y,  width_x,

          ps_y, ps_x, tile_x_with_halo,
          cpbase_y, cpend_y, index_y,
          tid_x, tid_y,
          stencilParaInput,
          sm_space, 0, cache_z-halo
        );
      }

    }
    if(!UseSMCache&&!UseRegCache)
    {
      for(int global_z=p_z; global_z<p_z_sm_end; global_z+=1)
      // for(int global_z=p_z; global_z<p_z_end; global_z+=1)
      {
        process_one_layer<REAL, halo, LOCAL_ITEM_PER_THREAD, LOCAL_TILE_X, LOCAL_TILE_Y, halo+1+isBOX, 2*halo+1>
        (
          input, output, 
          smbuffer_buffer_ptr, 
          r_smbuffer,

          global_z,  p_y,  p_x,
          width_z,  width_y,  width_x,

          ps_y, ps_x, tile_x_with_halo,
          cpbase_y, cpend_y, index_y,
          tid_x, tid_y,
          stencilParaInput
        ); 
      }
    }

    // general version
    // in(global, cache_sm_end+halo, p_z_end+halo) out(global, cache_sm_end, p_z_end) 
    for(int global_z=p_z_sm_end; global_z<p_z_end; global_z+=1)
    // for(int global_z=p_z; global_z<p_z_end; global_z+=1)
    {
      process_one_layer<REAL, halo, LOCAL_ITEM_PER_THREAD, LOCAL_TILE_X, LOCAL_TILE_Y, halo+1+isBOX, 2*halo+1>
      (
        input, output, 
        smbuffer_buffer_ptr, 
        r_smbuffer,

         global_z,  p_y,  p_x,
         width_z,  width_y,  width_x,

        ps_y, ps_x, tile_x_with_halo,
        cpbase_y, cpend_y, index_y,
        tid_x, tid_y,
        stencilParaInput
      ); 
    }
    
    //register east and west 
    _Pragma("unroll")
    for(int l_z=0; l_z<reg_folder_z; l_z++)
    {
      if(tid_x>=LOCAL_TILE_X-halo)
      {
        
        int l_x=tid_x-LOCAL_TILE_X+halo;
        // for(int l_y=threadIdx.x; l_y<LOCAL_TILE_Y;l_y+=blockDim.x)
        for(int l_y=0; l_y<ITEM_PER_THREAD;l_y+=1)
        {
          int local_y=l_y+LOCAL_ITEM_PER_THREAD*tid_y;
          //east
          boundary_buffer[boundary_east_step + (l_z) *  (LOCAL_TILE_Y+2*isBOX)*halo + (local_y+isBOX) + l_x * (LOCAL_TILE_Y+2*isBOX)]
          =
          r_space[l_z][l_y];
        }
      }

      if(tid_x<halo)
      {
        int l_x=tid_x;
        for(int l_y=0; l_y<ITEM_PER_THREAD;l_y+=1)
        {
          int local_y=l_y+LOCAL_ITEM_PER_THREAD*tid_y;
          //west
          boundary_buffer[boundary_west_step + (l_z) *  (LOCAL_TILE_Y+2*isBOX)*halo + (local_y+isBOX) + l_x * (LOCAL_TILE_Y+2*isBOX)]
          =
          r_space[l_z][l_y];
        }
      }
    }

    //sm east and west
    for(int l_z=0; l_z<max_sm_flder; l_z++)
    {
      for(int l_x=0; l_x<halo; l_x++)
      {
        for(int l_y=threadIdx.x; l_y<LOCAL_TILE_Y;l_y+=blockDim.x)
        {
          //east
          boundary_buffer[boundary_east_step + (l_z+reg_folder_z) *  (LOCAL_TILE_Y+2*isBOX)*halo + (l_y+isBOX) + l_x * (LOCAL_TILE_Y+2*isBOX)]
            =
            sm_space[l_z*(LOCAL_TILE_Y)*LOCAL_TILE_X+(l_y)*LOCAL_TILE_X+LOCAL_TILE_X-halo+l_x];

          //west
          boundary_buffer[boundary_west_step + (l_z+reg_folder_z) *  (LOCAL_TILE_Y+2*isBOX)*halo + (l_y+isBOX) + l_x * (LOCAL_TILE_Y+2*isBOX)]
            =
            sm_space[l_z*(LOCAL_TILE_Y)*LOCAL_TILE_X+(l_y)*LOCAL_TILE_X+l_x];
        }
      }
    }



    _Pragma("unroll")
    for(int l_z=0; l_z<reg_folder_z; l_z++)
    {
      
      int l_x=tid_x;
      #pragma unroll
      for(int l_y=0; l_y<halo; l_y++)
      {
        //south
        if(tid_y==0)
        {
          boundary_buffer[boundary_south_step + (l_z) *  (LOCAL_TILE_X)*halo + (l_x) + l_y * (LOCAL_TILE_X)]
            = r_space[l_z][l_y];
        }
        //north
        if(tid_y==dim_y-1)
        {
          boundary_buffer[boundary_north_step + (l_z) *  (LOCAL_TILE_X)*halo + (l_x) + (l_y) * (LOCAL_TILE_X)]
            = r_space[l_z][ITEM_PER_THREAD-halo+ l_y];
        } 
      }
    
    }

    //sm south and north
     _Pragma("unroll")
    for(int l_z=0; l_z<max_sm_flder; l_z++)
    {
      int l_x=tid_x;
      #pragma unroll
      for(int l_y=0; l_y<halo; l_y++)
      {
        //south
        if(tid_y==0)
        {
          boundary_buffer[boundary_south_step + (l_z+reg_folder_z) *  (LOCAL_TILE_X)*halo + (l_x) + l_y * (LOCAL_TILE_X)]
            = sm_space[l_z*(LOCAL_TILE_Y)*LOCAL_TILE_X+(l_y)*LOCAL_TILE_X+l_x];
        }
        //north
        if(tid_y==dim_y-1)
        {
          boundary_buffer[boundary_north_step + (l_z+reg_folder_z) *  (LOCAL_TILE_X)*halo + (l_x) + (l_y) * (LOCAL_TILE_X)]
            = sm_space[l_z*(LOCAL_TILE_Y)*LOCAL_TILE_X+(l_y+TILE_Y-halo)*LOCAL_TILE_X+l_x];
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

      // const int p_z =   .z * (blocksize_z) + (blockIdx.z<=z_quotient?blockIdx.z:z_quotient);

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
        for(int l_z=0; l_z<total_folder_z; l_z++)
        {
          //y
          for(int l_y=threadIdx.x; l_y<LOCAL_TILE_Y; l_y+=blockDim.x)
          {
            // //west
            l2_cache_o[l2_boundary_west_step + l_y + bid_y*LOCAL_TILE_Y + (l_x + bid_x*halo)*width_y + (l_z+bid_z*total_folder_z)*width_y*gdimx*halo ] 
            = boundary_buffer[boundary_west_step + (l_z) *  (LOCAL_TILE_Y+2*isBOX)*halo + (l_y+isBOX) + l_x * (LOCAL_TILE_Y+2*isBOX)];
            //east
            l2_cache_o[l2_boundary_east_step + l_y + bid_y*LOCAL_TILE_Y + (l_x + bid_x*halo)*width_y + (l_z+bid_z*total_folder_z)*width_y*gdimx*halo ]  
            = boundary_buffer[boundary_east_step + (l_z) *  (LOCAL_TILE_Y+2*isBOX)*halo + (l_y+isBOX) + l_x * (LOCAL_TILE_Y+2*isBOX)];
          }
        }
      }
      // // //north south
      for(int l_y=0; l_y<halo; l_y++)
      {
        //z
        // #pragma unroll
        // _Pragma("unroll")
        for(int l_z=0; l_z<total_folder_z; l_z++)
        {
          //y
          // #pragma unroll
          // _Pragma("unroll")
          for(int l_x=threadIdx.x; l_x<LOCAL_TILE_X; l_x+=blockDim.x)
          {
            // //north
            // l2_cache_o[((bid_x*2+0)*halo+l_x)*width_y*width_z + (l_z+p_z+1) * LOCAL_TILE_Y*dim_y + (l_y+bid_y*LOCAL_TILE_Y)]
            // l2_cache_o[ ((bid_x+bid_y*gdimx+ bid_z*gdimx*gdimy)*2+0)*halo*total_folder_z*LOCAL_TILE_Y + (l_z) * LOCAL_TILE_Y + (l_y) + l_x * boundary_step_yz] 
            l2_cache_o[l2_boundary_north_step + l_x + bid_x * TILE_X + (l_y + bid_y * halo) * width_x + (l_z + bid_z * total_folder_z) * width_x * gdimy*halo ] 
              = boundary_buffer[boundary_north_step + (l_z) *  (LOCAL_TILE_X)*halo + (l_x) + l_y * (LOCAL_TILE_X)];
            // south 
            l2_cache_o[l2_boundary_south_step + l_x + bid_x * TILE_X + (l_y + bid_y * halo) * width_x + (l_z + bid_z * total_folder_z) * width_x * gdimy*halo ]  
              = boundary_buffer[boundary_south_step + (l_z) *  (LOCAL_TILE_X)*halo + (l_x) + l_y * (LOCAL_TILE_X)];
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

      // int bid_y=blockIdx.x/gdim_x;
      int bid_x=blockIdx.x;
      int gdimx=gridDim.x;

      int bid_y=blockIdx.y;
      int gdimy=gridDim.y;
      
      int bid_z=blockIdx.z;
      //x
      for(int l_x=0; l_x<halo; l_x++)
      {
        //z
        for(int l_z=0; l_z<total_folder_z; l_z++)
        {
          //y
          for(int l_y=threadIdx.x-isBOX; l_y<LOCAL_TILE_Y+isBOX; l_y+=blockDim.x)
          {
            int l2_cache_l_y=MAX(l_y+ bid_y*LOCAL_TILE_Y,0);
            l2_cache_l_y=MIN(l2_cache_l_y,width_y-1);

            //east
            boundary_buffer[boundary_east_step + (l_z) *  (LOCAL_TILE_Y+2*isBOX)*halo+ (l_y+isBOX) + l_x * (LOCAL_TILE_Y+2*isBOX)] =
              (bid_x==gdimx-1?
              l2_cache_i[l2_boundary_east_step+ l2_cache_l_y  + ((halo-1) + bid_x*halo)*width_y + (l_z+bid_z*total_folder_z)*width_y*gdimx*halo ]  
             :
             l2_cache_i[l2_boundary_west_step+l2_cache_l_y  + (l_x + (bid_x+1)*halo)*width_y + (l_z+bid_z*total_folder_z)*width_y*gdimx*halo  ]
             )
             ;
            //west
            boundary_buffer[boundary_west_step + (l_z) *  (LOCAL_TILE_Y+2*isBOX)*halo + (l_y+isBOX) + l_x * (LOCAL_TILE_Y+2*isBOX)] =
              (bid_x==0?
                l2_cache_i[l2_boundary_west_step+ l2_cache_l_y  + (0 + bid_x*halo)*width_y + (l_z+bid_z*total_folder_z)*width_y*gdimx*halo ] 
                :
                l2_cache_i[l2_boundary_east_step+ l2_cache_l_y  + (l_x + (bid_x-1)*halo)*width_y + (l_z+bid_z*total_folder_z)*width_y*gdimx*halo ])
                ;
          }
        }
      }
      for(int l_y=0; l_y<halo; l_y++)
      {
        //z
        for(int l_z=0; l_z<total_folder_z; l_z++)
        {
          //y
          for(int l_x=threadIdx.x; l_x<LOCAL_TILE_X; l_x+=blockDim.x)
          {
            //north
            boundary_buffer[boundary_north_step + (l_z) *  (LOCAL_TILE_X)*halo + (l_x) + l_y * (LOCAL_TILE_X)] =
              (bid_y==gdimy-1?
             boundary_buffer[boundary_north_step + (l_z) *  (LOCAL_TILE_X)*halo + (l_x) + (halo-1) * (LOCAL_TILE_X)]
             :
             l2_cache_i[l2_boundary_south_step + l_x + bid_x * TILE_X + (l_y + (bid_y+1) * halo) * width_x + (l_z + bid_z * total_folder_z) * width_x * gdimy*halo  ]
             )
             ;
            //south
            boundary_buffer[boundary_south_step + (l_z) *  (LOCAL_TILE_X)*halo + (l_x) + l_y * (LOCAL_TILE_X)] =
              (bid_y==0?
                boundary_buffer[boundary_south_step + (l_z) *  (LOCAL_TILE_X)*halo + (l_x) + 0 * (LOCAL_TILE_X)]
                :
                l2_cache_i[l2_boundary_north_step + l_x + bid_x * TILE_X + (l_y + (bid_y-1) * halo) * width_x + (l_z + bid_z * total_folder_z) * width_x * gdimy*halo ])
                ;
          }
        }
      }
      __syncthreads();
    }
  }

  for(int global_z=p_z_reg_start, cache_z_reg=halo; global_z<p_z_sm_start; global_z+=1, cache_z_reg++)
  {
      _Pragma("unroll")
      for(int l_y=0; l_y<LOCAL_ITEM_PER_THREAD; l_y++)
      {
        int local_y=l_y+LOCAL_ITEM_PER_THREAD*tid_y;

        output[global_z*width_x*width_y+(p_y+local_y)*width_x+p_x+tid_x]  = r_space[cache_z_reg-halo][l_y];
          // sm_space[(cache_z-halo)*LOCAL_TILE_X*LOCAL_TILE_Y+(local_y-NOCACHE_Y)*LOCAL_TILE_X+tid_x];
      }  
      // break; 
  } 
  for(int global_z=p_z_sm_start, cache_z=halo; global_z<p_z_sm_end; global_z+=1, cache_z++)
  {
      _Pragma("unroll")
      for(int l_y=0; l_y<LOCAL_ITEM_PER_THREAD; l_y++)
      {
        int local_y=l_y+LOCAL_ITEM_PER_THREAD*tid_y;

        output[global_z*width_x*width_y+(p_y+local_y)*width_x+p_x+tid_x]  =
          sm_space[(cache_z-halo)*LOCAL_TILE_X*LOCAL_TILE_Y+(local_y)*LOCAL_TILE_X+tid_x];
      }  
      // break; 
  }
 
  #undef UseRegCache
}


// template __global__ void kernel3d_general<float,HALO,ITEM_PER_THREAD,TILE_X,TILE_Y,0,false> 
//     (float *__restrict__, float *__restrict__ , int , int , int, float*,float*,int,int);

// template __global__ void kernel3d_general<float,HALO,ITEM_PER_THREAD,TILE_X,TILE_Y,0,true> 
//     (float *__restrict__, float *__restrict__ , int , int , int, float*,float*,int,int);

// template __global__ void kernel3d_general<float,HALO,ITEM_PER_THREAD,TILE_X,TILE_Y,REG_FOLDER_Z,false> 
//     (float *__restrict__, float *__restrict__ , int , int , int, float*,float*,int,int);

template __global__ void kernel3d_general<float,HALO,ITEM_PER_THREAD,TILE_X,TILE_Y,REG_FOLDER_Z,true> 
    (float *__restrict__, float *__restrict__ , int , int , int, float*,float*,int,int);

    // template __global__ void kernel3d_general<double,HALO,LOCAL_ITEM_PER_THREAD,TILE_X,TILE_Y> 
//     (double *__restrict__, double *__restrict__ , int , int , int , double*,double*,int,int);

// #ifndef PERSISTENT 
//   PERKS_INITIALIZE_ALL_TYPE_1ARG(PERKS_DECLARE_INITIONIZATION_BASELINE,HALO);
// #else
//   PERKS_INITIALIZE_ALL_TYPE_1ARG(PERKS_DECLARE_INITIONIZATION_PERSISTENT,HALO);
// #endif