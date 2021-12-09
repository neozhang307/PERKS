// #include "./common/common.hpp"
// #include <cooperative_groups.h>
// #include <cuda.h>
// #include "stdio.h"
// #include "./common/cuda_computation.cuh"
// #include "./common/cuda_common.cuh"
// #include "./common/types.hpp"
#ifdef _TIMER_
#include "cuda_profiler_api.h"
#endif
#include <cuda.h>
#include "stdio.h"
#include <cooperative_groups.h>
#include "stdio.h"
#include "assert.h"
#include "config.cuh" 
#include "./common/jacobi_cuda.cuh"
#include "./common/types.hpp"
#include "./common/cuda_common.cuh"
#include "./common/cuda_computation.cuh"

#define TILE_X 256
#define NAIVE

#define cudaCheckError() {                                          \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
 }                                                                 \
}


__global__ void printptx(int *result)
{
  // printf("code is run in %d\n",PERKS_ARCH);
  result[0]=PERKS_ARCH;
}
void host_printptx(int&result)
{
  int*d_r;
  cudaMalloc((void**)&d_r, sizeof(int));
  printptx<<<1,1>>>(d_r);
  cudaMemcpy(&result, d_r, sizeof(int), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
}


template<class REAL>
void j3d_iterative(REAL * h_input, int height, int width_y, int width_x, REAL * __var_0__, int iteration){
  // int iteration=4;
/* Host allocation Begin */
  int sm_count;
  cudaDeviceGetAttribute ( &sm_count, cudaDevAttrMultiProcessorCount,0 );
#ifndef __PRINT__
  printf("sm_count is %d\n",sm_count);
#endif

#ifndef NAIVE
    int sharememory0=((TILE_Y+2*HALO)*(TILE_X+2*HALO)*(1+2*HALO)+1)*sizeof(REAL);
    #ifndef __PRINT__
        printf("shared memroy size is %f KB\n",(REAL)sharememory0/1024);
    #endif
#endif
#if defined(GEN) || defined(MIX)
    int sharememory1 = sharememory0+2*BD_STEP_XY*FOLDER_Z*sizeof(REAL);
    int sharememory2 = sharememory1 + sizeof(REAL) * (SFOLDER_Z)*(TILE_Y*2-1)*TILE_X;
#endif

  REAL * input;
  cudaMalloc(&input,sizeof(REAL)*(height*width_x*width_y));
  Check_CUDA_Error("Allocation Error!! : input\n");
  // cudaPointerAttributes ptrAttrib_h_input;
  // cudaMemcpyKind memcpy_kind_h_input = cudaMemcpyHostToDevice;
  // if (cudaPointerGetAttributes(&ptrAttrib_h_input, h_input) == cudaSuccess)
    // if (ptrAttrib_h_input.memoryType == cudaMemoryTypeDevice)
      // memcpy_kind_h_input = cudaMemcpyDeviceToDevice;
  cudaGetLastError();
  // if( memcpy_kind_h_input != cudaMemcpyDeviceToDevice ){
  cudaMemcpy(input,h_input,sizeof(REAL)*(height*width_x*width_y), cudaMemcpyHostToDevice);
  // }
  REAL * __var_1__;
  cudaMalloc(&__var_1__,sizeof(REAL)*(height*width_x*width_y));
  Check_CUDA_Error("Allocation Error!! : __var_1__\n");
  REAL * __var_2__;
  cudaMalloc(&__var_2__,sizeof(REAL)*(height*width_x*width_y));
  Check_CUDA_Error("Allocation Error!! : __var_2__\n");

#ifndef NAIVE
  int maxSharedMemory;
  cudaDeviceGetAttribute (&maxSharedMemory, cudaDevAttrMaxSharedMemoryPerMultiprocessor,0 );
  int SharedMemoryUsed=maxSharedMemory-1024;
  #ifndef __PRINT__  
    printf("MAX shared memory is %f KB but only use %f KB\n",maxSharedMemory/1024.0,SharedMemoryUsed/1024.0);
  #endif
#endif

#ifdef PERSISTENT
    cudaFuncSetAttribute(kernel_persistent_iterative, cudaFuncAttributeMaxDynamicSharedMemorySize, SharedMemoryUsed);
#endif
#ifdef GEN
    // cudaFuncSetAttribute(kernel_persistent_iterative_register, cudaFuncAttributeMaxDynamicSharedMemorySize, SharedMemoryUsed);
    cudaFuncSetAttribute(kernel_persistent_iterative_gen, cudaFuncAttributeMaxDynamicSharedMemorySize, SharedMemoryUsed);
#endif
#ifdef BASEPER
    cudaFuncSetAttribute(kernel_persistent, cudaFuncAttributeMaxDynamicSharedMemorySize, SharedMemoryUsed);
#endif
#ifdef BASELINE
    cudaFuncSetAttribute(kernel_baseline_2, cudaFuncAttributeMaxDynamicSharedMemorySize, SharedMemoryUsed);
#endif


/*Host Allocation End */
/* Kernel Launch Begin */
#ifdef NAIVE
  dim3 block_dim_1(TILE_X, 4, 1);
  dim3 grid_dim_1(width_x/TILE_X, width_y/4, height);
#endif
#ifdef BASELINE
  dim3 block_dim2(TILE_X, 1, 1);
  dim3 grid_dim2(width_x/TILE_X, width_y/TILE_Y,  (sm_count*8*1024/TILE_X)/(width_x/TILE_X)/(width_y/TILE_Y));
#endif
#if defined(GEN) || defined(PERSISTENT)||defined(BASEPER) 
  int numBlocksPerSm_current=0;
  #ifdef GEN
    if(SFOLDER_Z==0)
    {
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksPerSm_current, kernel_persistent_iterative_gen, TILE_X, sharememory1);
    }
    else
    {
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksPerSm_current, kernel_persistent_iterative_gen, TILE_X, sharememory2);
    }
  #endif

  #if defined(BASEPER)
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksPerSm_current, kernel_persistent, TILE_X, sharememory0);
  #endif
  #ifdef PERSISTENT
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksPerSm_current, kernel_persistent_iterative, TILE_X, sharememory0);
  #endif

  dim3 block_dim3(TILE_X, 1, 1);
  // dim3 grid_dim3(GDIM_X, 1, GDIM_Z);
  // numBlocksPerSm_current=MIN(2,numBlocksPerSm_current);
  dim3 grid_dim3(MIN(width_x*width_y/TILE_X/TILE_Y,sm_count*numBlocksPerSm_current), 1, sm_count*numBlocksPerSm_current/MIN(width_x*width_y/TILE_X/TILE_Y,sm_count*numBlocksPerSm_current));
  #ifndef __PRINT__
    printf("blk/sm is %d\n",numBlocksPerSm_current);
  #endif
#endif
  cudaDeviceSynchronize();
  cudaCheckError();


  size_t L2_utage = width_y*height*sizeof(REAL)*HALO*(width_x/TILE_X)*2 ;
#ifndef __PRINT__
  printf("l2 cache used is %ld KB : 4096 KB \n",L2_utage/1024);
#endif
  REAL * L2_cache;
  REAL * L2_cache1;
  cudaMalloc(&L2_cache,L2_utage);
  cudaMalloc(&L2_cache1,L2_utage);
   
#if defined(GEN) || defined(MIX)||defined(PERSISTENT) 
int l_iteration=iteration;
#endif
#ifdef PERSISTENT
void* KernelArgs0[] ={(void**)&input,(void*)&__var_2__,
    (void**)&height,(void**)&width_y,(void*)&width_x,
    (void*)&l_iteration};
#ifdef __PRINT__  
void* KernelArgs0NULL[] ={(void**)&__var_2__,(void*)&__var_1__,
    (void**)&height,(void**)&width_y,(void*)&width_x,
    (void*)&l_iteration};
#endif
#endif
#ifdef GEN
  void* KernelArgs1[] ={(void**)&input,(void*)&__var_2__,
    (void**)&height,(void**)&width_y,(void*)&width_x,
    (void**)&L2_cache,(void**)&L2_cache1,
    (void*)&l_iteration};
#ifdef __PRINT__  
  void* KernelArgs1NULL[] ={(void**)&__var_2__,(void*)&__var_1__,
    (void**)&height,(void**)&width_y,(void*)&width_x,
    (void**)&L2_cache,(void**)&L2_cache1,
    (void*)&l_iteration};
#endif 
#endif
#ifndef __PRINT__
  // printf("shared memroy size is %f KB\n",(REAL)sharememory0/1024);
#ifndef NAIVE
  printf("shared memroy size is %f KB\n",(REAL)sharememory0/1024);
#endif
#if defined(GEN) || defined(MIX)
  printf("shared memroy size (add boundary) is %f KB\n",(REAL)sharememory1/1024);
  printf("shared memroy size when use shared memory to cache(add boundary) is %f KB\n",(REAL)sharememory2/1024);
#endif
#endif
#ifdef __PRINT__
  #ifdef NAIVE
    printf("naive\t");
  #endif 
  #ifdef BASELINE
    #ifndef DA100X
      printf("bsln\t");
    #else
      printf("asyncbsln\t");
    #endif
  #endif 
  #ifdef PERSISTENT
    #ifndef DA100X
      printf("psstnt\t");
    #else
      printf("asyncpsstnt\t");
    #endif
  #endif 
  #ifdef GEN
    #ifndef DA100X
      printf("gen"); 
    #else
      printf("asyncgen"); 
    #endif
    #if RFOLDER_Z==0 && SFOLDER_Z ==0
      printf("\t");
    #endif
    #if RFOLDER_Z==0 && SFOLDER_Z !=0
      printf("_sm\t");
    #endif
    #if RFOLDER_Z!=0 && SFOLDER_Z ==0
      printf("_reg\t");
    #endif
    #if RFOLDER_Z!=0 && SFOLDER_Z !=0
      printf("_mix\t");
    #endif
  #endif
  #ifdef BASEPER
    #ifndef DA100X
      printf("psbsln\t");
    #else
      printf("asyncpsbsln\t");
    #endif
  #endif
  printf("%d\t%d\t%d\t%d\t",height,width_y,width_x,iteration); 
  #ifdef NAIVE
    printf("(%d,%d,%d)\t",TILE_X,4,1); 
    printf("(%d,%d,%d)\t%d\t0\t0\t",grid_dim_1.x, grid_dim_1.y, grid_dim_1.z ,grid_dim_1.x*grid_dim_1.y*grid_dim_1.z/sm_count,grid_dim_1.x*grid_dim_1.y*grid_dim_1.z/sm_count);
      kernel3d_restrict<REAL,HALO><<<grid_dim_1, block_dim_1>>>
            (__var_2__, __var_1__,  height, width_y, width_x);
  #endif 
  #ifdef BASELINE
    printf("(%d,%d,%d)\t",TILE_X,1,1); 
    printf("(%d,%d,%d)\t%d\t0\t0\t",grid_dim2.x, grid_dim2.y, grid_dim2.z, grid_dim2.x*grid_dim2.y*grid_dim2.z/sm_count ,grid_dim2.x*grid_dim2.y*grid_dim2.z/sm_count);
       kernel_baseline_2<<<grid_dim2, block_dim2, sharememory0>>>
          (input, __var_2__, height, width_y, width_x);
  #endif 
  #ifdef BASEPER
    printf("(%d,%d,%d)\t",TILE_X,1,1); 
    printf("(%d,%d,%d)\t%d\t0\t0\t",grid_dim3.x, grid_dim3.y, grid_dim3.z ,grid_dim3.x*grid_dim3.y*grid_dim3.z/sm_count ,grid_dim3.x*grid_dim3.y*grid_dim3.z/sm_count);

    kernel_persistent<<<grid_dim3, block_dim3, sharememory0>>>
          (input, __var_2__, height, width_y, width_x);
  #endif
  #ifdef PERSISTENT
    printf("(%d,%d,%d)\t",TILE_X,1,1 );
    printf("(%d,%d,%d)\t%d\t0\t0\t",grid_dim3.x, grid_dim3.y, grid_dim3.z ,grid_dim3.x*grid_dim3.y*grid_dim3.z/sm_count);

    cudaLaunchCooperativeKernel((void*)kernel_persistent_iterative, grid_dim3, block_dim3, KernelArgs0NULL, sharememory0,0);
  #endif 
  #ifdef GEN
    printf("(%d,%d,%d)\t",TILE_X,1,1);
    printf("(%d,%d,%d)\t%d\t%d\t%d\t",grid_dim3.x, grid_dim3.y, grid_dim3.z,grid_dim3.x*grid_dim3.y*grid_dim3.z/sm_count,RFOLDER_Z,SFOLDER_Z);
    if(SFOLDER_Z==0)
    {
      cudaLaunchCooperativeKernel((void*)kernel_persistent_iterative_gen, grid_dim3, block_dim3, KernelArgs1NULL, sharememory1,0);
    }
    else
    {
      cudaLaunchCooperativeKernel((void*)kernel_persistent_iterative_gen, grid_dim3, block_dim3, KernelArgs1NULL, sharememory2,0);
    }
  #endif
#endif


#ifdef _TIMER_
  cudaEvent_t _forma_timer_start_,_forma_timer_stop_;
  cudaEventCreate(&_forma_timer_start_);
  cudaEventCreate(&_forma_timer_stop_);
  cudaEventRecord(_forma_timer_start_,0);
#endif

// #define PERSISTENT
  
#ifdef BASELINE 
#ifndef __PRINT__
  printf("baseline is executed\n");
#endif
  cudaDeviceSynchronize();
  cudaCheckError();
  kernel_baseline_2<<<grid_dim2, block_dim2, sharememory0>>>
          (input, __var_2__, height, width_y, width_x);

  for(int i=1; i<iteration; i++)
  {
     kernel_baseline_2<<<grid_dim2, block_dim2, sharememory0>>>
          (__var_2__, __var_1__, height, width_y, width_x);
    REAL* tmp = __var_2__;
    __var_2__=__var_1__;
    __var_1__= tmp;
  }

  cudaDeviceSynchronize();
  cudaCheckError();
#endif


  // // int sharememroy1 = sharememory0+(TILE_Y)
#ifdef BASEPER

  // printf("%d\n",GDIM_X);
  kernel_persistent<<<grid_dim3, block_dim3, sharememory0>>>
          (input, __var_2__, height, width_y, width_x);

  for(int i=1; i<iteration; i++)
  {
     kernel_persistent<<<grid_dim3, block_dim3, sharememory0>>>
          (__var_2__, __var_1__, height, width_y, width_x);
    REAL* tmp = __var_2__;
    __var_2__=__var_1__;
    __var_1__= tmp;
  }
#endif
  //persistent kernel
// #define GEN

#ifdef PERSISTENT
  cudaLaunchCooperativeKernel((void*)kernel_persistent_iterative, grid_dim3, block_dim3, KernelArgs0, sharememory0,0);
#endif
  //cached persistent kernel

#ifdef GEN

  // cudaFuncSetAttribute(kernel_persistent_iterative_register, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024);
  // cudaFuncSetAttribute(kernel_persistent_iterative_gen, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024);

  if(SFOLDER_Z==0)
  {
    #ifndef __PRINT__
      printf("execute register only version\n");
    #endif
    cudaLaunchCooperativeKernel((void*)kernel_persistent_iterative_gen, grid_dim3, block_dim3, KernelArgs1, sharememory1,0);
  }
  else
  {
    // printf("execute here 1\n");
    cudaLaunchCooperativeKernel((void*)kernel_persistent_iterative_gen, grid_dim3, block_dim3, KernelArgs1, sharememory2,0);
  }

#endif



#ifdef NAIVE
// //naive

  kernel3d_restrict<REAL,HALO><<<grid_dim_1, block_dim_1>>>
          (input, __var_2__,  height, width_y, width_x);

  for(int i=1; i<iteration; i++)
  {
     kernel3d_restrict<REAL,HALO><<<grid_dim_1, block_dim_1>>>
          (__var_2__, __var_1__, height, width_y, width_x);
    REAL* tmp = __var_2__;
    __var_2__=__var_1__;
    __var_1__= tmp;
  }

#endif
#ifdef CHECK
  cudaDeviceSynchronize();
  cudaCheckError();
#endif
#ifdef _TIMER_
  cudaEventRecord(_forma_timer_stop_,0);
  cudaEventSynchronize(_forma_timer_stop_);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime,_forma_timer_start_,_forma_timer_stop_);
#ifndef __PRINT__
  printf("[FORMA] SIZE : %d,%d,%d\n",height,width_y,width_x);
  printf("[FORMA] Computation Time(ms) : %lf\n",elapsedTime);
  printf("[FORMA] Speed(GCells/s) : %lf\n",(REAL)iteration*height*width_x*width_y/ elapsedTime/1000/1000);
  printf("[FORMA] Computation(GFLOPS/s) : %lf\n",(REAL)iteration*height*width_x*width_y*(HALO*2+1)*(HALO*2+1)/ elapsedTime/1000/1000);
  printf("[FORMA] Bandwidht(GB/s) : %lf\n",(REAL)iteration*height*width_x*width_y*sizeof(REAL)*2/ elapsedTime/1000/1000);
#else
  //h y x iter TILEX thready=1 gridx gridy latency speed 
  // printf("%d\t%d\t%d\t%d\t",height,width_y,width_x,iteration); 
  // printf("%d\t%d\t",TILE_X,1); 
  // printf("%d\t%d\t%d\t",width_x/TILE_X, width_y/TILE_Y, 5); 
  printf("%f\t%lf\n",elapsedTime,(REAL)iteration*height*width_x*width_y/ elapsedTime/1000/1000); 
  printf("");
#endif
  cudaEventDestroy(_forma_timer_start_);
  cudaEventDestroy(_forma_timer_stop_);
#endif
  cudaDeviceSynchronize();
  cudaCheckError();
  
#if defined(GEN) || defined(PERSISTENT)
if(iteration%2==1)  
{
  cudaMemcpy(__var_0__, __var_2__, sizeof(REAL)*height*width_x*width_y, cudaMemcpyDeviceToHost);
}
else
{
  cudaMemcpy(__var_0__, input, sizeof(REAL)*height*width_x*width_y, cudaMemcpyDeviceToHost);
}
#else
  cudaMemcpy(__var_0__, __var_2__, sizeof(REAL)*height*width_x*width_y, cudaMemcpyDeviceToHost);
#endif
  cudaDeviceSynchronize();
  cudaCheckError();

  cudaFree(input);
  cudaFree(__var_1__);
  cudaFree(__var_2__);
  cudaFree(L2_cache);
  cudaFree(L2_cache1);
}

template void j3d_iterative<float>(float * h_input, int height, int width_y, int width_x, float * __var_0__, int iteration);
