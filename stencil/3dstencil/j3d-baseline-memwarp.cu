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


// template<class REAL, int halo, int BASE_Z, int SIZE_Z, int SMSIZE, int SM_BASE=0, bool isInit=false, bool sync=true>
// __device__ void __forceinline__ global2sm(REAL *src, REAL* smbuffer_buffer_ptr[SMSIZE],
//                                           int gbase_x, int gbase_y, int gbase_z,
//                                           int width_x, int width_y, int width_z,
//                                           int sm_width_x, int sm_base_x,
//                                           int size_y, int sm_base_y, int ind_y,
//                                           int tid_x)
// {
//   //

template<class REAL, int halo, int LOCAL_ITEM_PER_THREAD, int LOCAL_TILE_X, int LOCAL_TILE_Y>
__global__ void kernel3d_baseline_memwarp(REAL * __restrict__ input, 
                                REAL * __restrict__ output, 
                                int width_z, int width_y, int width_x) 
{
  // printf("?");
  const int tile_x_with_halo=LOCAL_TILE_X+2*halo;
  const int tile_y_with_halo=LOCAL_TILE_Y+2*halo;
  stencilParaT;
  
  extern __shared__ char sm[];
  REAL* sm_rbuffer = (REAL*)sm+1;

  register REAL r_smbuffer[2*halo+1][LOCAL_ITEM_PER_THREAD];
  // printf("%d\n",LOCAL_ITEM_PER_THREAD);
  // return;
  REAL* smbuffer_buffer_ptr[halo+1];
  smbuffer_buffer_ptr[0]=sm_rbuffer;
  #pragma unroll
  for(int hl=1; hl<halo+1; hl++)
  {
    smbuffer_buffer_ptr[hl]=smbuffer_buffer_ptr[hl-1]+tile_x_with_halo*tile_y_with_halo;
  }

  const int tid_x = threadIdx.x%LOCAL_TILE_X;
  const int tid_y = threadIdx.x/LOCAL_TILE_X-2;

  const int index_y=tid_y*LOCAL_ITEM_PER_THREAD;

  const int ps_y = halo;
  const int ps_x = halo;
  const int ps_z = halo;

  const int p_x = blockIdx.x * LOCAL_TILE_X;
  const int p_y = blockIdx.y * LOCAL_TILE_Y;
  // if(blockIdx.x==0&&tid_x==0)
  //   printf("<%d,%d,%d:%d,%d>",ps_x,ps_y,ps_z,tid_x,tid_y);
  // return;   32*32
  int blocksize_z=(width_z/gridDim.z);
  int z_quotient = width_z%gridDim.z;

  const int p_z =  blockIdx.z * (blocksize_z) + (blockIdx.z<=z_quotient?blockIdx.z:z_quotient);
  blocksize_z += (blockIdx.z<z_quotient?1:0);
  const int p_z_end = p_z + (blocksize_z);
 
  // int smz_ind=0;
  if(tid_y>=0)
  {
    // glb2reg 
    _Pragma("unroll")
    for(int l_y=0; l_y<LOCAL_ITEM_PER_THREAD; l_y++)
    {
      _Pragma("unroll")
      for(int l_z=-halo; l_z<1+halo ; l_z++)
      {
        int l_global_z = (MIN(p_z+l_z,width_z-1));
          l_global_z = (MAX(l_global_z,0));
        int l_global_y = (MIN(p_y+l_y+index_y,width_y-1));
          l_global_y = (MAX(l_global_y,0));

        r_smbuffer[l_z+ps_z][l_y] = input[l_global_z*width_x*width_y+l_global_y*width_x+
              ((p_x+tid_x))];
      }
    }

    _Pragma("unroll")
    for(int l_z=0; l_z<halo; l_z++)
    {
      int l_global_z = (MAX(p_z+l_z+0,0));
          l_global_z = (MIN(l_global_z,width_x-1));
      _Pragma("unroll")
      for(int l_y=0; l_y<LOCAL_ITEM_PER_THREAD; l_y+=1)
      {
        int l_global_y = (MIN(p_y+l_y+index_y,width_y-1));
          l_global_y = (MAX(l_global_y,0));
          smbuffer_buffer_ptr[l_z+0][tile_x_with_halo*(l_y+ps_y+index_y) + (tid_x) + ps_x]=
              input[l_global_z*width_x*width_y+l_global_y*width_x+
              MAX((p_x+tid_x),0)];
      }
    }

    for(int global_z=p_z; global_z<p_z_end; global_z+=1)
    {
      __syncthreads();
      
       _Pragma("unroll")
      for(int l_z=0; l_z<1; l_z++)
      {
        int l_global_z = (MAX(global_z+l_z+halo,0));
            l_global_z = (MIN(l_global_z,width_x-1));
        _Pragma("unroll")
        for(int l_y=0; l_y<LOCAL_ITEM_PER_THREAD; l_y+=1)
        {
          int l_global_y = (MIN(p_y+l_y+index_y,width_y-1));
            l_global_y = (MAX(l_global_y,0));
            smbuffer_buffer_ptr[l_z+halo][tile_x_with_halo*(l_y+ps_y+index_y) + (tid_x) + ps_x]=
                input[l_global_z*width_x*width_y+l_global_y*width_x+
                MAX((p_x+tid_x),0)];
          // if(tid_x<halo*2)
          // {
          //     smbuffer_buffer_ptr[l_z+halo][tile_x_with_halo*(l_y+ps_y+index_y) + tid_x + LOCAL_TILE_X-halo+ps_x]=
          //         input[l_global_z*width_x*width_y+l_global_y*width_x+
          //           MIN(p_x+tid_x-halo+LOCAL_TILE_X,width_x-1)];
          // }
        }
      }
      
      __syncthreads();
      
      REAL sum[LOCAL_ITEM_PER_THREAD];

      //sm2reg
      _Pragma("unroll")
      for(int l_y=0; l_y<LOCAL_ITEM_PER_THREAD; l_y++)
      {
      _Pragma("unroll")
        for(int l_z=halo; l_z<1+halo ; l_z++)
        { 
           r_smbuffer[l_z+ps_z][l_y] = 
            smbuffer_buffer_ptr[(l_z)][tile_x_with_halo*(l_y+ps_y+index_y) + tid_x+ps_x];
        }
      }

      // #pragma unroll
      _Pragma("unroll")
      for(int l_y=0; l_y<LOCAL_ITEM_PER_THREAD; l_y++)
      {
        sum[l_y]=0;
      }

      //main computation
      computation<REAL,LOCAL_ITEM_PER_THREAD,halo>( sum,
                                      smbuffer_buffer_ptr,
                                      ps_y+index_y, tile_x_with_halo, tid_x+ps_x,
                                      r_smbuffer,
                                      stencilParaInput);

      // reg 2 ptr
      _Pragma("unroll")
      for(int l_y=0; l_y<LOCAL_ITEM_PER_THREAD; l_y++)
      {
        output[(global_z)*width_x*width_y+(l_y+p_y+index_y)*width_x+
                  (p_x+tid_x)]= sum[l_y];
      }

      REAL* tmp = smbuffer_buffer_ptr[0];
      // sm2sm
      _Pragma("unroll")
      for(int hl=1; hl<halo+1; hl++)
      {
        smbuffer_buffer_ptr[hl-1]=smbuffer_buffer_ptr[hl];
      }
      smbuffer_buffer_ptr[halo]=tmp;

      // reg2reg
      _Pragma("unroll")
      for(int l_y=0; l_y<LOCAL_ITEM_PER_THREAD; l_y++)
      {
        _Pragma("unroll")
        for(int l_z=-halo; l_z<halo ; l_z++)
        { 
          r_smbuffer[l_z+ps_z][l_y] = r_smbuffer[l_z+ps_z+1][l_y];
        }
      }
    }
  }
  else if(tid_y==-1)
  {
    //north south
    const int index_y=0;
    const int index_y_end=LOCAL_TILE_Y;
    _Pragma("unroll")
    for(int l_z=0; l_z<halo; l_z++)
    {
      int l_global_z = (MAX(p_z+l_z+0,0));
          l_global_z = (MIN(l_global_z,width_x-1));
      _Pragma("unroll")
      for(int l_y=-halo; l_y<0; l_y+=1)
      {
        int l_global_y = (MIN(p_y+l_y+index_y,width_y-1));
          l_global_y = (MAX(l_global_y,0));
          smbuffer_buffer_ptr[l_z+0][tile_x_with_halo*(l_y+ps_y+index_y) + (tid_x-halo) + ps_x]=
              input[l_global_z*width_x*width_y+l_global_y*width_x+
              MAX((p_x+tid_x-halo),0)];
        if(tid_x<halo*2)
        {
            smbuffer_buffer_ptr[l_z+0][tile_x_with_halo*(l_y+ps_y+index_y) + tid_x + LOCAL_TILE_X-halo+ps_x]=
                input[l_global_z*width_x*width_y+l_global_y*width_x+
                  MIN(p_x+tid_x-halo+LOCAL_TILE_X,width_x-1)];
        }
      }

      _Pragma("unroll")
      for(int l_y=0; l_y<halo; l_y+=1)
      {
        int l_global_y = (MIN(p_y+l_y+index_y_end,width_y-1));
          l_global_y = (MAX(l_global_y,0));
          smbuffer_buffer_ptr[l_z+0][tile_x_with_halo*(l_y+ps_y+index_y_end) + (tid_x-halo) + ps_x]=
              input[l_global_z*width_x*width_y+l_global_y*width_x+
              MAX((p_x+tid_x-halo),0)];
        if(tid_x<halo*2)
        {
            smbuffer_buffer_ptr[l_z+0][tile_x_with_halo*(l_y+ps_y+index_y_end) + tid_x + LOCAL_TILE_X-halo+ps_x]=
                input[l_global_z*width_x*width_y+l_global_y*width_x+
                  MIN(p_x+tid_x-halo+LOCAL_TILE_X,width_x-1)];
        }
      }
    }

    for(int global_z=p_z; global_z<p_z_end; global_z+=1)
    {
      __syncthreads();
      
       _Pragma("unroll")
      for(int l_z=0; l_z<1; l_z++)
      {
        int l_global_z = (MAX(global_z+l_z+halo,0));
            l_global_z = (MIN(l_global_z,width_x-1));
        _Pragma("unroll")
        for(int l_y=-halo; l_y<0; l_y+=1)
        {
          int l_global_y = (MIN(p_y+l_y+index_y,width_y-1));
            l_global_y = (MAX(l_global_y,0));
            smbuffer_buffer_ptr[l_z+halo][tile_x_with_halo*(l_y+ps_y+index_y) + (tid_x-halo) + ps_x]=
                input[l_global_z*width_x*width_y+l_global_y*width_x+
                MAX((p_x+tid_x-halo),0)];
          if(tid_x<halo*2)
          {
              smbuffer_buffer_ptr[l_z+halo][tile_x_with_halo*(l_y+ps_y+index_y) + tid_x + LOCAL_TILE_X-halo+ps_x]=
                  input[l_global_z*width_x*width_y+l_global_y*width_x+
                    MIN(p_x+tid_x-halo+LOCAL_TILE_X,width_x-1)];
          }
        }

        _Pragma("unroll")
        for(int l_y=0; l_y<halo; l_y+=1)
        {
          int l_global_y = (MIN(p_y+l_y+index_y_end,width_y-1));
            l_global_y = (MAX(l_global_y,0));
            smbuffer_buffer_ptr[l_z+halo][tile_x_with_halo*(l_y+ps_y+index_y_end) + (tid_x-halo) + ps_x]=
                input[l_global_z*width_x*width_y+l_global_y*width_x+
                MAX((p_x+tid_x-halo),0)];
          if(tid_x<halo*2)
          {
              smbuffer_buffer_ptr[l_z+halo][tile_x_with_halo*(l_y+ps_y+index_y_end) + tid_x + LOCAL_TILE_X-halo+ps_x]=
                  input[l_global_z*width_x*width_y+l_global_y*width_x+
                    MIN(p_x+tid_x-halo+LOCAL_TILE_X,width_x-1)];
          }
        }
      }
      __syncthreads();
    
      REAL* tmp = smbuffer_buffer_ptr[0];
      // sm2sm
      _Pragma("unroll")
      for(int hl=1; hl<halo+1; hl++)
      {
        smbuffer_buffer_ptr[hl-1]=smbuffer_buffer_ptr[hl];
      }
      smbuffer_buffer_ptr[halo]=tmp;
    }
  }
  else if(tid_y==-2)
  { //east west 
    // const int index_y=0;
    // const int index_y_end=LOCAL_TILE_Y;
    _Pragma("unroll")
    for(int l_z=0; l_z<halo; l_z++)
    {
      int l_global_z = (MAX(p_z+l_z+0,0));
          l_global_z = (MIN(l_global_z,width_x-1));
      // _Pragma("unroll")
      for(int l_y=tid_x; l_y<LOCAL_TILE_Y; l_y+=LOCAL_TILE_X)
      {
        //west
        _Pragma("unroll")
        for(int l_x=-halo; l_x<0; l_x++)
        {
          int l_global_x = (MIN(p_x+l_x,width_x-1));
          l_global_x = (MAX(l_global_x,0));
          smbuffer_buffer_ptr[l_z+0][tile_x_with_halo*(l_y+ps_y) + l_x + ps_x]=
              input[l_global_z*width_x*width_y+(p_y+l_y)*width_x+
              l_global_x];
        }
        //east
         _Pragma("unroll")
        for(int l_x=0; l_x<halo; l_x++)
        {
          int l_global_x = (MIN(p_x+l_x+LOCAL_TILE_X,width_x-1));
          l_global_x = (MAX(l_global_x,0));
          smbuffer_buffer_ptr[l_z+0][tile_x_with_halo*(l_y+ps_y) + l_x+LOCAL_TILE_X + ps_x]=
              input[l_global_z*width_x*width_y+(p_y+l_y)*width_x+
              l_global_x];
        }
      }
    }

    for(int global_z=p_z; global_z<p_z_end; global_z+=1)
    {
      __syncthreads();
      
       _Pragma("unroll")
      for(int l_z=0; l_z<1; l_z++)
      {
        int l_global_z = (MAX(global_z+l_z+halo,0));
            l_global_z = (MIN(l_global_z,width_x-1));
        for(int l_y=tid_x; l_y<LOCAL_TILE_Y; l_y+=LOCAL_TILE_X)
        {
          //west
          _Pragma("unroll")
          for(int l_x=-halo; l_x<0; l_x++)
          {
            int l_global_x = (MIN(p_x+l_x,width_x-1));
            l_global_x = (MAX(l_global_x,0));
            smbuffer_buffer_ptr[l_z+halo][tile_x_with_halo*(l_y+ps_y) + l_x + ps_x]=
                input[l_global_z*width_x*width_y+(p_y+l_y)*width_x+
                l_global_x];
          }
        // }
          //east
          _Pragma("unroll")
          for(int l_x=0; l_x<halo; l_x++)
          {
            int l_global_x = (MIN(p_x+l_x+LOCAL_TILE_X,width_x-1));
            l_global_x = (MAX(l_global_x,0));
            smbuffer_buffer_ptr[l_z+halo][tile_x_with_halo*(l_y+ps_y) + l_x + LOCAL_TILE_X+ ps_x]=
                input[l_global_z*width_x*width_y+(p_y+l_y)*width_x+
                l_global_x];
          }
        }
      }
      __syncthreads();
    
      REAL* tmp = smbuffer_buffer_ptr[0];
      // sm2sm
      _Pragma("unroll")
      for(int hl=1; hl<halo+1; hl++)
      {
        smbuffer_buffer_ptr[hl-1]=smbuffer_buffer_ptr[hl];
      }
      smbuffer_buffer_ptr[halo]=tmp;
    }
  }
}


template __global__ void kernel3d_baseline_memwarp<float,HALO,ITEM_PER_THREAD,TILE_X,TILE_Y> 
    (float *__restrict__, float *__restrict__ , int , int , int );
template __global__ void kernel3d_baseline_memwarp<double,HALO,ITEM_PER_THREAD,TILE_X,TILE_Y> 
    (double *__restrict__, double *__restrict__ , int , int , int );