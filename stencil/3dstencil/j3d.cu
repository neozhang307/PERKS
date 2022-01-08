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

// #define TILE_X 256
// #define NAIVE
#if defined(NAIVE)||defined(BASELINE)||defined(BASELINE_CM)
  #define TRADITIONLAUNCH
#endif
#if defined(GEN)|| defined(GENWR) ||defined(PERSISTENT)
  #define PERSISTENTLAUNCH
#endif
#if defined PERSISTENTLAUNCH||defined(BASELINE_CM)
  #define PERSISTENTTHREAD
#endif
#if defined(BASELINE)||defined(BASELINE_MEMWARP)||defined(BASELINE_CM) ||defined(GEN)||defined(GENWR)||defined(PERSISTENT)
  #define USEMAXSM
#endif


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

// template<class REAL, int halo>
// __global__ void kernel3d_baseline(REAL *  input, 
//                                 REAL *  output, 
//                                 int width_z, int width_y, int width_x) 
// {
//   printf("??");
// }
template<class REAL>
void j3d_iterative(REAL * h_input, int height, int width_y, int width_x, REAL * __var_0__, int iteration){
  // int iteration=4;
/* Host allocation Begin */
  int sm_count;
  cudaDeviceGetAttribute ( &sm_count, cudaDevAttrMultiProcessorCount,0 );
// #ifndef __PRINT__
  printf("sm_count is %d\n",sm_count);
// #endif

  int ptx;
  host_printptx(ptx);
  printf("code is run in %d\n",ptx);
#ifdef NAIVE
  auto execute_kernel = kernel3d_restrict<REAL,HALO>;
#endif 
#if defined(BASELINE) ||defined(BASELINE_CM)
  auto execute_kernel = kernel3d_baseline<REAL,HALO>;
#endif
#ifdef BASELINE_MEMWARP
  auto execute_kernel = kernel3d_baseline_memwarp<REAL,HALO>;
#endif
//shared memory related 
size_t executeSM=0;
#ifndef NAIVE
    int basic_sm_space=((TILE_Y+2*HALO)*(TILE_X+2*HALO)*(1+2*HALO)+1)*sizeof(REAL);
    executeSM=basic_sm_space;
#endif
printf("sm is %ld\n",executeSM);
#if defined(GEN) || defined(MIX)
    int sharememory1 = basic_sm_space+2*BD_STEP_XY*FOLDER_Z*sizeof(REAL);
    int sharememory2 = sharememory1 + sizeof(REAL) * (SFOLDER_Z)*(TILE_Y*2-1)*TILE_X;
#endif

  REAL * input;
  cudaMalloc(&input,sizeof(REAL)*(height*width_x*width_y));
  Check_CUDA_Error("Allocation Error!! : input\n");

  cudaGetLastError();
  cudaMemcpy(input,h_input,sizeof(REAL)*(height*width_x*width_y), cudaMemcpyHostToDevice);
  REAL * __var_1__;
  cudaMalloc(&__var_1__,sizeof(REAL)*(height*width_x*width_y));
  Check_CUDA_Error("Allocation Error!! : __var_1__\n");
  REAL * __var_2__;
  cudaMalloc(&__var_2__,sizeof(REAL)*(height*width_x*width_y));
  Check_CUDA_Error("Allocation Error!! : __var_2__\n");

#ifdef USEMAXSM
  int maxSharedMemory;
  cudaDeviceGetAttribute (&maxSharedMemory, cudaDevAttrMaxSharedMemoryPerMultiprocessor,0 );
  int SharedMemoryUsed=maxSharedMemory-1024;
  cudaFuncSetAttribute(execute_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SharedMemoryUsed);
#endif 

/*Host Allocation End */
/* Kernel Launch Begin */
// #ifndef

#ifdef NAIVE
  dim3 block_dim_1(bdimx, 4, 1);
  dim3 grid_dim_1(width_x/bdimx, width_y/4, height);

  dim3 executeBlockDim=block_dim_1;
  dim3 executeGridDim=grid_dim_1;

#endif
#ifdef BASELINE
  dim3 block_dim_2(bdimx, 1, 1);
  dim3 grid_dim_2(width_x/TILE_X, width_y/TILE_Y, max(2,(sm_count*8)*TILE_X*TILE_Y/width_x/width_y));
  // dim3 block_dim3(TILE_X, 1, 1);
  // dim3 grid_dim3(MIN(width_x*width_y/TILE_X/TILE_Y,sm_count*numBlocksPerSm_current), 1, sm_count*numBlocksPerSm_current/MIN(width_x*width_y/TILE_X/TILE_Y,sm_count*numBlocksPerSm_current));
  
  printf("<%d,%d,%d>",grid_dim_2.x,grid_dim_2.y,grid_dim_2.z);
  dim3 executeBlockDim=block_dim_2;
  dim3 executeGridDim=grid_dim_2;
#endif
#ifdef BASELINE_MEMWARP
  dim3 block_dim_2(bdimx+2*TILE_X, 1, 1);
  dim3 grid_dim_2(width_x/TILE_X, width_y/TILE_Y,max(2,(sm_count*8)*TILE_X*TILE_Y/width_x/width_y));
  // dim3 block_dim3(TILE_X, 1, 1);
  // dim3 grid_dim3(MIN(width_x*width_y/TILE_X/TILE_Y,sm_count*numBlocksPerSm_current), 1, sm_count*numBlocksPerSm_current/MIN(width_x*width_y/TILE_X/TILE_Y,sm_count*numBlocksPerSm_current));
  
  printf("<%d,%d,%d>",grid_dim_2.x,grid_dim_2.y,grid_dim_2.z);
  dim3 executeBlockDim=block_dim_2;
  dim3 executeGridDim=grid_dim_2;
#endif
#if defined(PERSISTENTTHREAD)
  int numBlocksPerSm_current=0;

  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksPerSm_current, execute_kernel, bdimx, executeSM);

  dim3 block_dim_3(bdimx, 1, 1);
  dim3 grid_dim_3(width_x/TILE_X, width_y/TILE_Y, MAX(1,sm_count*numBlocksPerSm_current/(width_x*width_y/TILE_X/TILE_Y)));
  dim3 executeBlockDim=block_dim_3;
  dim3 executeGridDim=grid_dim_3;

#endif


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
// #ifndef __PRINT__
//   // printf("shared memroy size is %f KB\n",(REAL)sharememory0/1024);
//   #ifndef NAIVE
//     printf("shared memroy size is %f KB\n",(REAL)sharememory0/1024);
//   #endif
//   #if defined(GEN) || defined(MIX)
//     printf("shared memroy size (add boundary) is %f KB\n",(REAL)sharememory1/1024);
//     printf("shared memroy size when use shared memory to cache(add boundary) is %f KB\n",(REAL)sharememory2/1024);
//   #endif
// #endif
bool warmup=false;
if(warmup)
{
  #ifdef TRADITIONLAUNCH
    execute_kernel<<<executeGridDim, executeGridDim, executeSM>>>
            (__var_2__, __var_1__,  height, width_y, width_x);
  #endif
}

#ifdef _TIMER_
  cudaEvent_t _forma_timer_start_,_forma_timer_stop_;
  cudaEventCreate(&_forma_timer_start_);
  cudaEventCreate(&_forma_timer_stop_);
  cudaEventRecord(_forma_timer_start_,0);
#endif

// #define PERSISTENT

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

  cudaDeviceSynchronize();
  cudaCheckError();
#ifdef TRADITIONLAUNCH

  execute_kernel<<<executeGridDim, executeBlockDim,executeSM>>>
          (input, __var_2__,  height, width_y, width_x);
  cudaDeviceSynchronize();
  cudaCheckError();
  for(int i=1; i<iteration; i++)
  {
     execute_kernel<<<executeGridDim, executeBlockDim,executeSM>>>
          (__var_2__, __var_1__, height, width_y, width_x);
    REAL* tmp = __var_2__;
    __var_2__=__var_1__;
    __var_1__= tmp;
  }

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
  // printf("");
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
