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
//   // _Pragma("unroll")
//   // for(int l_z=0; l_z<SIZE_Z; l_z++)
//   // {
//   //   int l_global_z = (MAX(gbase_z+l_z+BASE_Z,0));
//   //       l_global_z = (MIN(l_global_z,width_x-1));
//   //   _Pragma("unroll")
//   //   for(int l_y=0; l_y<size_y; l_y+=1)
//   //   {
//   //     int l_global_y = (MIN(gbase_y+l_y-halo,width_y-1));
//   //       l_global_y = (MAX(l_global_y,0));
//   //       smbuffer_buffer_ptr[l_z+BASE_Z][sm_width_x*(l_y+sm_base_y) + (tid_x-halo) + sm_base_x]=
//   //           src[l_global_z*width_x*width_y+l_global_y*width_x+
//   //           MAX((gbase_x+tid_x-halo),0)];
//   //     if(tid_x<halo*2)
//   //     {
//   //         smbuffer_buffer_ptr[l_z+BASE_Z][sm_width_x*(l_y+sm_base_y) + tid_x + blockDim.x-halo+sm_base_x]=
//   //             src[l_global_z*width_x*width_y+l_global_y*width_x+
//   //               MIN(gbase_x+tid_x-halo+blockDim.x,width_x-1)];
//   //     }
//   //   }
//   // }
//   // if(sync)
//   // {
//   //   __syncthreads();
//   // }

//   _Pragma("unroll")
//   for(int l_z=0; l_z<SIZE_Z; l_z++)
//   {
//     // int l_global_z = (MIN(p_z+l_z,width_z-1));
//         // l_global_z = (MAX(l_global_z,0));
//     int l_global_z = (MAX(gbase_z+l_z+BASE_Z,0));
//         l_global_z = (MIN(l_global_z,width_z-1));
//     // #pragma unroll
//     _Pragma("unroll")
//     for(int l_y=0; l_y<size_y; l_y+=1)
//     {
//       int l_global_y = (MIN(gbase_y+l_y+ind_y,width_y-1));
//         l_global_y = (MAX(l_global_y,0));
//       #ifndef ASYNCSM
//         smbuffer_buffer_ptr[l_z+BASE_Z][sm_width_x*(l_y+sm_base_y+ind_y) + (tid_x-halo) + sm_base_x]=
//             src[l_global_z*width_x*width_y+l_global_y*width_x+
//             MAX((gbase_x+tid_x-halo),0)];
//       #else
//         __pipeline_memcpy_async(smbuffer_buffer_ptr[l_z+BASE_Z] + sm_width_x*(l_y+sm_base_y+ind_y) + (tid_x-halo) + sm_base_x, 
//           src + l_global_z*width_x*width_y+l_global_y*width_x+
//               MAX((gbase_x+tid_x-halo),0)
//           , sizeof(REAL));
//       #endif
//       if(tid_x<halo*2)
//       {
//         // sm_rbuffer[ SM_X*SM_Y*((l_z+ps_z+smz_ind)%(Halo*2+1)) + SM_X*(l_y+ps_y) + tid_x + blockDim.x-Halo+ps_x]=
//         #ifndef ASYNCSM
//           smbuffer_buffer_ptr[l_z+BASE_Z][sm_width_x*(l_y+sm_base_y+ind_y) + tid_x + TILE_X-halo+sm_base_x]=
//               src[l_global_z*width_x*width_y+l_global_y*width_x+
//                 MIN(gbase_x+tid_x-halo+TILE_X,width_x-1)];
//         #else
//           __pipeline_memcpy_async(smbuffer_buffer_ptr[l_z+BASE_Z] + sm_width_x*(l_y+sm_base_y+ind_y) + tid_x + TILE_X -halo+sm_base_x, 
//               src + l_global_z*width_x*width_y+l_global_y*width_x+
//                   MIN(gbase_x+tid_x-halo+TILE_X,width_x-1)
//               , sizeof(REAL));
//         #endif
//       }
//     }
//   }
//   #ifdef ASYNCSM
//     __pipeline_commit();
//   #endif
//   if(sync)
//   {
//     #ifdef ASYNCSM
//       __pipeline_wait_prior(0);
//     #endif
//     __syncthreads();
//   }
// }

template<class REAL, int halo>
__global__ void kernel3d_baseline(REAL * __restrict__ input, 
                                REAL * __restrict__ output, 
                                int width_z, int width_y, int width_x) 
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

  const int ps_y = halo + ITEM_PER_THREAD*tid_y;
  const int ps_x = halo;
  const int ps_z = halo;

  const int p_x = blockIdx.x * TILE_X;
  const int p_y = blockIdx.y * TILE_Y + ITEM_PER_THREAD*tid_y;
  // if(blockIdx.x==0&&tid_x==0)
  //   printf("<%d,%d,%d:%d,%d>",ps_x,ps_y,ps_z,tid_x,tid_y);
  // return;
  int blocksize_z=(width_z/gridDim.z);
  int z_quotient = width_z%gridDim.z;

  const int p_z =  blockIdx.z * (blocksize_z) + (blockIdx.z<=z_quotient?blockIdx.z:z_quotient);
  blocksize_z += (blockIdx.z<z_quotient?1:0);
  const int p_z_end = p_z + (blocksize_z);
 
  // int smz_ind=0;

  {
    //glb2reg 
    _Pragma("unroll")
    for(int l_y=0; l_y<ITEM_PER_THREAD; l_y++)
    {
      _Pragma("unroll")
      for(int l_z=-halo; l_z<1+halo ; l_z++)
      {
        int l_global_z = (MIN(p_z+l_z,width_z-1));
          l_global_z = (MAX(l_global_z,0));
        int l_global_y = (MIN(p_y+l_y,width_y-1));
          l_global_y = (MAX(l_global_y,0));

        r_smbuffer[l_z+ps_z][l_y] = input[l_global_z*width_x*width_y+l_global_y*width_x+
              ((p_x+tid_x))];
      }
    }
    global2sm<REAL, halo, 0, halo, halo+1, 0, true, false>
                                        (input, smbuffer_buffer_ptr,
                                          p_x, p_y, p_z,
                                          width_x, width_y, width_z,

                                          tile_x_with_halo, ps_x,
                                          ITEM_PER_THREAD+2*halo, ps_y,-halo, 
                                          TILE_X, tid_x);
    for(int global_z=p_z; global_z<p_z_end; global_z+=1)
    {
      __syncthreads();
      
      global2sm<REAL, halo, halo, 1, halo+1, 0, false, true>
                                        (input, smbuffer_buffer_ptr,
                                          p_x, p_y, global_z,
                                          width_x, width_y, width_z,

                                          tile_x_with_halo, ps_x,
                                          ITEM_PER_THREAD+2*halo, ps_y,-halo, 
                                          TILE_X,tid_x);
      __syncthreads();
      
      REAL sum[ITEM_PER_THREAD];

      //sm2reg
      _Pragma("unroll")
      for(int l_y=0; l_y<ITEM_PER_THREAD; l_y++)
      {
      _Pragma("unroll")
        for(int l_z=halo; l_z<1+halo ; l_z++)
        { 
           r_smbuffer[l_z+ps_z][l_y] = 
            smbuffer_buffer_ptr[(l_z)][tile_x_with_halo*(l_y+ps_y) + tid_x+ps_x];
        }
      }

      // #pragma unroll
      _Pragma("unroll")
      for(int l_y=0; l_y<ITEM_PER_THREAD; l_y++)
      {
        sum[l_y]=0;
      }

      //main computation
      computation<REAL,ITEM_PER_THREAD,halo>( sum,
                                      smbuffer_buffer_ptr[0],
                                      ps_y, tile_x_with_halo, tid_x+ps_x,
                                      r_smbuffer,
                                      stencilParaInput);

      // reg 2 ptr
      _Pragma("unroll")
      for(int l_y=0; l_y<ITEM_PER_THREAD; l_y++)
      {
        output[(global_z)*width_x*width_y+(l_y+p_y)*width_x+
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
      for(int l_y=0; l_y<ITEM_PER_THREAD; l_y++)
      {
        _Pragma("unroll")
        for(int l_z=-halo; l_z<halo ; l_z++)
        { 
          r_smbuffer[l_z+ps_z][l_y] = r_smbuffer[l_z+ps_z+1][l_y];
        }
      }
    }
  }
}


template __global__ void kernel3d_baseline<float,HALO> 
    (float *__restrict__, float *__restrict__ , int , int , int );
template __global__ void kernel3d_baseline<double,HALO> 
    (double *__restrict__, double *__restrict__ , int , int , int );