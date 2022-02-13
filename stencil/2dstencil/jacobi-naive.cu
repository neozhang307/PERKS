#include "./config.cuh"
#include "./common/types.hpp"
#include "./common/cuda_computation.cuh"
#include "./common/cuda_common.cuh"
#include <math.h>

template<class REAL, int halo>
__global__ void 
#ifndef BOX
kernel2d_restrict
#else
kernel2d_restrict_box
#endif
        (REAL* input,int width_y, int width_x, REAL* output) 
{
  stencilParaT;
  int l_x = blockDim.x * blockIdx.x + threadIdx.x;  
  int l_y = blockDim.y * blockIdx.y + threadIdx.y;

#ifndef BOX
  int c = l_x + l_y * width_x;
  // int w[halo];
  // int e[halo];
  // int n[halo];
  // int s[halo];
  // // _Pragma("unroll") 
  // for(int hl=0; hl<halo; hl++)
  // {
  //   w[hl] = max(0,l_x-1-hl)+l_y * width_x;
  //   e[hl] = min(width_x-1,l_x+1+hl)+l_y * width_x;
  //   s[hl] = l_x+max(0,l_y-1-hl) * width_x;;
  //   n[hl] = l_x+min(width_y-1,l_y+1+hl) * width_x;
  // }
  REAL sum=0;
  // _Pragma("unroll") 
#ifndef NAIVENVCC
  #pragma unroll 1
#endif
  for(int hl=0; hl<halo; hl++)
  {
    // sum+=south[hl]*input[s[hl]];
    sum+=south[hl]*input[l_x+max(0,l_y-1-hl) * width_x];
  }
  // _Pragma("unroll") 
#ifndef NAIVENVCC
  #pragma unroll 1
#endif
  for(int hl=0; hl<halo; hl++)
  {
    // sum+=west[hl]*input[w[hl]];
    sum+=west[hl]*input[max(0,l_x-1-hl)+l_y * width_x];
  }
  sum+=center*input[c];
  // _Pragma("unroll") 
#ifndef NAIVENVCC
  #pragma unroll 1
#endif
  for(int hl=0; hl<halo; hl++)
  {
    // sum+=east[hl]*input[e[hl]];
    sum+=east[hl]*input[min(width_x-1,l_x+1+hl)+l_y * width_x];
  }
  // _Pragma("unroll") 
#ifndef NAIVENVCC
  #pragma unroll 1
#endif
  for(int hl=0; hl<halo; hl++)
  {
    // sum+=north[hl]*input[n[hl]];
    sum+=north[hl]*input[l_x+min(width_y-1,l_y+1+hl) * width_x];
  }
  output[c]=sum;
  return;
#else
  // int vertical[HALO*2+1];
  // int horizontal[HALO*2+1];
  // #pragma unroll
  // for(int hl_y=-HALO; hl_y<=HALO; hl_y++)
  // {
  //   vertical[hl_y+HALO]=min(max(l_y+hl_y,0),width_y-1)*width_x;
  // }
  // #pragma unroll
  // for(int hl_x=-HALO; hl_x<=HALO; hl_x++)
  // {
  //   horizontal[hl_x+HALO]=min(max(l_x+hl_x,0),width_x-1);
  // }
  REAL sum=0;
  // #pragma unroll
#ifndef NAIVENVCC
  #pragma unroll 1
#endif
  for(int hl_y=-HALO; hl_y<=HALO; hl_y++)
  {
#ifndef NAIVENVCC
  #pragma unroll 1
#endif
    for(int hl_x=-HALO; hl_x<=HALO; hl_x++)
    {
      sum+=filter[hl_y+HALO][hl_x+HALO]*
      input[min(max(l_y+hl_y,0),width_y-1)*width_x  + min(max(l_x+hl_x,0),width_x-1)];
    }
  }
  output[l_y*width_x  + l_x]=sum;
  return;
#endif
}

// template __global__ void kernel2d_restrict<double,1>(double*,int,int,double*);
// template __global__ void kernel2d_restrict<float,1>(float*,int,int,float*);
#ifndef BOX
PERKS_INITIALIZE_ALL_TYPE_1ARG(PERKS_DECLARE_INITIONIZATION_REFERENCE,HALO);
#else
PERKS_INITIALIZE_ALL_TYPE_1ARG(PERKS_DECLARE_INITIONIZATION_REFERENCE_BOX,HALO);
#endif
// PERKS_INITIALIZE_ALL_TYPE_WITH_HALO(PERKS_DECLARE_INITIONIZATION_REFERENCE,2);