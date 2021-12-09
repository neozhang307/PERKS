#include "./config.cuh"
#include "./common/cuda_computation.cuh"
#include "./common/types.hpp"
#include "./common/cuda_common.cuh"


template<class REAL, int halo>
__global__ void kernel3d_restrict(REAL* input, REAL* output,
                                  int height, int width_y, int width_x) 
{
    stencilParaT;

    int l_x = blockDim.x * blockIdx.x + threadIdx.x;  
    int l_y = blockDim.y * blockIdx.y + threadIdx.y;
    int l_h = blockDim.z * blockIdx.z + threadIdx.z;
#ifndef BOX
    int c  = l_x + l_y * width_x + l_h * width_x*width_y;
    int w[halo];
    int e[halo];
    int n[halo];
    int s[halo];
    int t[halo];
    int b[halo];
    #pragma unroll 
    for(int hl=0; hl<halo; hl++)
    {
        w[hl] = MAX(l_x-1-hl, 0) + l_y * width_x +  l_h*width_x*width_y;
        e[hl] = MIN(width_x-1,l_x+1+hl)+  l_y * width_x +  l_h*width_x*width_y;
        s[hl] = l_x+MAX(0,l_y-1-hl) * width_x +  l_h*width_x*width_y;
        n[hl] = l_x+MIN(width_y-1,l_y+1+hl) * width_x +  l_h*width_x*width_y;
        b[hl] = l_x + l_y * width_x  +  MAX(0,l_h-1-hl)*width_x*width_y;
        t[hl] = l_x + l_y * width_x  +  MIN(height-1,l_h+1+hl)*width_x*width_y;
    }
    REAL sum=0;
    #pragma unroll 
    for(int hl=0; hl<halo; hl++)
    {
        sum+=south[halo-1-hl]*input[s[halo-1-hl]];
    }
    #pragma unroll 
    for(int hl=0; hl<halo; hl++)
    {
        sum+=west[halo-1-hl]*input[w[halo-1-hl]];
    }
    sum+=center*input[c];
    #pragma unroll 
    for(int hl=0; hl<halo; hl++)
    {
        sum+=east[halo-1-hl]*input[e[halo-1-hl]];
    }
    #pragma unroll 
    for(int hl=0; hl<halo; hl++)
    {
        sum+=north[halo-1-hl]*input[n[halo-1-hl]];
    }

    #pragma unroll 
    for(int hl=0; hl<halo; hl++)
    {
        sum+=bottom[halo-1-hl]*input[b[halo-1-hl]];
    }
    #pragma unroll 
    for(int hl=0; hl<halo; hl++)
    {
        sum+=top[halo-1-hl]*input[t[halo-1-hl]];
    }
    output[c]=sum;
#else
    int x_axle[halo*2+1];
    int y_axle[halo*2+1];
    int h_axle[halo*2+1];
    #pragma unroll
    for(int hl_h=-halo; hl_h<=halo; hl_h++)
    {
        h_axle[hl_h+halo]=MIN(MAX(l_h+hl_h,0),height-1)*width_x*width_y;
    }
    #pragma unroll
    for(int hl_y=-halo; hl_y<=halo; hl_y++)
    {
        y_axle[hl_y+halo]=MIN(MAX(l_y+hl_y,0),width_y-1)*width_x;
    }
    #pragma unroll
    for(int hl_x=-halo; hl_x<=halo; hl_x++)
    {
        x_axle[hl_x+halo]=MIN(MAX(l_x+hl_x,0),width_x-1);
    }

    REAL sum=0;
    // #pragma unroll
    for(int hl_h=-halo; hl_h<=halo; hl_h++)
    {
        // int hl_h=0;
        #pragma unroll
        for(int hl_y=-halo; hl_y<=halo; hl_y++)
        {
            #pragma unroll 
            for(int hl_x=-halo; hl_x<=halo; hl_x++)
            {
                sum+=filter[hl_h+halo][hl_y+halo][hl_x+halo]==0?0:filter[hl_h+halo][hl_y+halo][hl_x+halo]*input[h_axle[hl_h+halo] + y_axle[hl_y+halo] + x_axle[hl_x+halo]];
            }
        }
    }
    output[h_axle[halo] + y_axle[halo] + x_axle[halo]]=sum;
#endif

    return;
}

template __global__ void kernel3d_restrict<float,HALO>(float* input, float* output,
                                  int height, int width_y, int width_x);