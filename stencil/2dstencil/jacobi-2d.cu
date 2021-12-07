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



#ifdef GEN
#include "./genconfig.cuh"
#endif

//#ifndef REAL
//#define REAL float
//#endif
//configuration
#if defined(NAIVE)||defined(BASELINE)||defined(BASELINE_CM)
  #define TRADITIONLAUNCH
#endif
#if defined(GEN)||defined(MIX)||defined(PERSISTENT)
  #define PERSISTENTLAUNCH
#endif
#if defined PERSISTENTLAUNCH||defined(BASELINE_CM)
  #define PERSISTENTTHREAD
#endif
#if defined(BASELINE)||defined(BASELINE_CM)||defined(GEN)||defined(MIX)||defined(PERSISTENT)
  #define USEMAXSM
#endif

#ifdef __PRINT__ 
  #define WARMUPRUN
#endif


namespace cg = cooperative_groups;


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

#ifndef RTILE_Y
#define RTILE_Y (8)
#endif
#ifndef TILE_X
#define TILE_X (256)
#endif

#define bdim_x (TILE_X)

#define BASIC_TILE_X (TILE_X+2*HALO)
#define BASIC_TILE_Y (RTILE_Y+2*HALO)
#define BASIC_SM_SPACE (BASIC_TILE_X)*(BASIC_TILE_Y)


#define TOTAL_SM_TILE_Y (RTILE_Y*SM_FOLER_Y)
#define TOTAL_REG_TILE_Y (RTILE_Y*REG_FOLDER_Y)
#define TOTAL_SM_CACHE_SPACE (TILE_X+2*HALO)*(TOTAL_SM_TILE_Y+2*HALO)

#define TILE_Y (TOTAL_SM_TILE_Y+TOTAL_REG_TILE_Y)




template<class REAL>
// void jacobi_iterative(REAL * h_input, int width_y, int width_x, REAL * __var_0__, int iteration, bool async=false){
void jacobi_iterative(REAL * h_input, int width_y, int width_x, REAL * __var_0__, int iteration){
// extern "C" void jacobi_iterative(REAL * h_input, int width_y, int width_x, REAL * __var_0__, int iteration){
/* Host allocation Begin */
  int ptx;
  host_printptx(ptx);
  #ifndef __PRINT__
    printf("code is run in %d\n",ptx);
  #endif
/*************************************/
  // if(ptx<800&&async==true)printf("error async not support\n");//lower ptw not support 

//initialization
#if defined(PERSISTENT)
  #ifndef BOX
  auto execute_kernel = kernel_persistent_baseline<REAL,RTILE_Y,HALO>;
  #else
  auto execute_kernel = kernel_persistent_baseline_box<REAL,RTILE_Y,HALO>;
  #endif
#endif
#if defined(BASELINE_CM)||defined(BASELINE)
  #ifndef BOX
    auto execute_kernel = kernel_baseline<REAL,RTILE_Y,HALO>;
  #else
    auto execute_kernel = kernel_baseline_box<REAL,RTILE_Y,HALO>;
  #endif
#endif
#ifdef NAIVE
  #ifndef BOX
    auto execute_kernel = kernel2d_restrict<REAL,HALO>;
  #else
    auto execute_kernel = kernel2d_restrict_box<REAL,HALO>;
  #endif
#endif 
#ifdef GEN
  #ifndef BOX
  auto execute_kernel = kernel_general<REAL,RTILE_Y,HALO,REG_FOLDER_Y,true>;
  #else
  auto execute_kernel = kernel_general_box<REAL,RTILE_Y,HALO,REG_FOLDER_Y,true>;
  #endif
  //auto execute_kernel = kernel_general<REAL,RTILE_Y,HALO,REG_FOLDER_Y,UseSMCache>;
#endif
  int sm_count;
  cudaDeviceGetAttribute ( &sm_count, cudaDevAttrMultiProcessorCount,0 );
  
  //initialization input and output space
  REAL * input;
  cudaMalloc(&input,sizeof(REAL)*((width_y-0)*(width_x-0)));
  Check_CUDA_Error("Allocation Error!! : input\n");
  cudaMemcpy(input,h_input,sizeof(REAL)*((width_y-0)*(width_x-0)), cudaMemcpyHostToDevice);
  REAL * __var_1__;
  cudaMalloc(&__var_1__,sizeof(REAL)*((width_y-0)*(width_x-0)));
  Check_CUDA_Error("Allocation Error!! : __var_1__\n");
  REAL * __var_2__;
  cudaMalloc(&__var_2__,sizeof(REAL)*((width_y-0)*(width_x-0)));
  Check_CUDA_Error("Allocation Error!! : __var_2__\n");

  //initialize tmp space for halo region
#if defined(GEN) || defined(MIX)|| defined(PERSISTENT)
  REAL * L2_cache3;
  REAL * L2_cache4;
  size_t L2_utage_2 = sizeof(REAL)*(width_y)*2*(width_x/bdim_x)*HALO;
#ifndef __PRINT__
  printf("l2 cache used is %ld KB : 4096 KB \n",L2_utage_2*2/1024);
#endif
  cudaMalloc(&L2_cache3,L2_utage_2*2);
  L2_cache4=L2_cache3+(width_y)*2*(width_x/bdim_x)*HALO;
#endif

  //initialize shared memory
  int maxSharedMemory;
  cudaDeviceGetAttribute (&maxSharedMemory, cudaDevAttrMaxSharedMemoryPerMultiprocessor,0 );
  //could not use all share memory in a100. so set it in default.
  int SharedMemoryUsed = maxSharedMemory-1024;

#if defined(USEMAXSM)
  cudaFuncSetAttribute(execute_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SharedMemoryUsed);
#endif

size_t executeSM = 0;
#ifndef NAIVE
  //shared memory used for compuation
  int basic_sm_space=(RTILE_Y+2*HALO)*(TILE_X+2*HALO);
  size_t sharememory_basic=(1+basic_sm_space)*sizeof(REAL);
  executeSM = sharememory_basic;

#endif

  #ifdef PERSISTENT
    size_t max_sm_flder=0;
  #endif 

  #define halo HALO
  #if defined(GEN) || defined(MIX)
  size_t max_sm_flder=0;
  max_sm_flder=(SharedMemoryUsed/sizeof(REAL)
                          -2*HALO*isBOX
                          -basic_sm_space
                          -2*HALO*(REG_FOLDER_Y)*RTILE_Y
                          -2*HALO*(TILE_X+2*HALO))/(TILE_X+4*HALO)/RTILE_Y;

  // size_t sm_cache_size = TOTAL_SM_CACHE_SPACE*sizeof(REAL);
  size_t sm_cache_size = (max_sm_flder*RTILE_Y+2*HALO)*(TILE_X+2*HALO)*sizeof(REAL);
  size_t y_axle_halo = (HALO*2*((max_sm_flder + REG_FOLDER_Y)*RTILE_Y+isBOX))*sizeof(REAL);
  executeSM=sharememory_basic+y_axle_halo;
  executeSM+=sm_cache_size;
  #undef halo
#ifndef __PRINT__
  printf("the max flder is %ld and the total sm size is %ld\n", max_sm_flder, executeSM);
#endif

  //size_t sharememory3=sharememory_basic+(HALO*2*(TILE_Y))*sizeof(REAL);
  //size_t sharememory4=sharememory3-(STILE_SIZE*sizeof(REAL));
#endif


#ifdef PERSISTENTTHREAD
  int numBlocksPerSm_current=0;

  #ifdef MIX
    if(SM_FOLER_Y!=0)
    {
      // cudaLaunchCooperativeKernel((void*)kernel_mix, grid_dim, block_dim, KernelArgs2,sharememory3,0);
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksPerSm_current, kernel_mix, bdim_x, sharememory3);
    }
    else
    {
      // cudaLaunchCooperativeKernel((void*)kernel_mix_reg, grid_dim, block_dim, KernelArgs2,sharememory4,0);
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksPerSm_current, kernel_mix_reg, bdim_x, sharememory4);
    }
  
  #endif
  #if defined(BASELINE_CM)||defined(PERSISTENT)||defined(GEN)
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksPerSm_current, execute_kernel, bdim_x, executeSM);
  #endif

  dim3 block_dim(bdim_x);
  dim3 grid_dim(width_x/bdim_x,sm_count*numBlocksPerSm_current/(width_x/bdim_x));
  
  dim3 executeBlockDim=block_dim;
  dim3 executeGridDim=grid_dim;
#endif 
#ifdef NAIVE
  dim3 block_dim_1(MIN(width_x,bdim_x),1);
  dim3 grid_dim_1(width_x/MIN(width_x,bdim_x),width_y/1);

  dim3 executeBlockDim=block_dim_1;
  dim3 executeGridDim=grid_dim_1;
#endif
#ifdef BASELINE
  dim3 block_dim2(bdim_x);
  dim3 grid_dim2(width_x/bdim_x,MIN((sm_count*8*1024/bdim_x)/(width_x/bdim_x),width_y/RTILE_Y));
//  printf("<%d,%d,%d>",); 
  dim3 executeBlockDim=block_dim2;
  dim3 executeGridDim=grid_dim2;

#endif
//in order to get a better performance, warmup run is necessary.

#ifdef MIX
  int l_iteration=iteration;
  void* KernelArgs2[] ={(void**)&input,(void**)&width_y,
    (void*)&width_x,(void*)&__var_2__,(void*)&L2_cache1,(void*)&L2_cache1,
    (void*)&l_iteration};
#endif

#if defined(GEN) || defined(PERSISTENT)
  int l_iteration=iteration;
  void* ExecuteKernelArgs[] ={(void**)&input,(void**)&width_y,
    (void*)&width_x,(void*)&__var_2__,(void*)&L2_cache3,(void*)&L2_cache4,
    (void*)&l_iteration, (void*)&max_sm_flder};

  #ifdef WARMUPRUN
    void* KernelArgs_NULL[] ={(void**)&__var_2__,(void**)&width_y,
      (void*)&width_x,(void*)&__var_1__,(void*)&L2_cache3,(void*)&L2_cache4,
      (void*)&l_iteration, (void *)&max_sm_flder};
  #endif

#endif

#if defined(GEN) && defined(L2PER)
    // REAL l2perused;
    // size_t inner_window_size = 30*1024*1024;
    // cudaStreamAttrValue stream_attribute;
    // stream_attribute.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(L2_cache3);                  // Global Memory data pointer
    // stream_attribute.accessPolicyWindow.num_bytes = min(inner_window_size,L2_utage_2*2);                                   // Number of bytes for persistence access
    // stream_attribute.accessPolicyWindow.hitRatio  = 1;                                             // Hint for cache hit ratio
    // stream_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;                  // Persistence Property
    // stream_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;  

    // cudaStreamSetAttribute(0, cudaStreamAttributeAccessPolicyWindow, &stream_attribute); 
    // cudaCtxResetPersistingL2Cache();
    // cudaStreamSynchronize(0);
#endif

#ifdef WARMUPRUN
      cudaCheckError();
  #ifdef TRADITIONLAUNCH
      execute_kernel<<<executeGridDim, executeBlockDim, executeSM>>>
            (__var_2__, width_y, width_x,__var_1__);
  #endif 

  #ifdef PERSISTENTLAUNCH
      cudaLaunchCooperativeKernel((void*)execute_kernel, executeGridDim, executeBlockDim, KernelArgs_NULL, executeSM,0);
  #endif

#endif 

#ifdef _TIMER_
  cudaEvent_t _forma_timer_start_,_forma_timer_stop_;
  cudaEventCreate(&_forma_timer_start_);
  cudaEventCreate(&_forma_timer_stop_);
  cudaEventRecord(_forma_timer_start_,0);
#endif
#ifdef MIX
  if(SM_FOLER_Y!=0)
  {
    cudaLaunchCooperativeKernel((void*)kernel_mix, grid_dim, block_dim, KernelArgs2,sharememory3,0);
  }
  else
  {
    cudaLaunchCooperativeKernel((void*)kernel_mix_reg, grid_dim, block_dim, KernelArgs2,sharememory4,0);
  }
#endif
#ifdef PERSISTENTLAUNCH
  cudaLaunchCooperativeKernel((void*)execute_kernel, 
            executeGridDim, executeBlockDim, 
            ExecuteKernelArgs, 
            //KernelArgs4,
            executeSM,0);
#endif
#ifdef TRADITIONLAUNCH
  execute_kernel<<<executeGridDim, executeBlockDim, executeSM>>>
          (input, width_y, width_x, __var_2__);

  for(int i=1; i<iteration; i++)
  {
     execute_kernel<<<executeGridDim, executeBlockDim, executeSM>>>
          (__var_2__, width_y, width_x , __var_1__);
    REAL* tmp = __var_2__;
    __var_2__=__var_1__;
    __var_1__= tmp;
  }
  cudaCheckError();
#endif


#ifdef CHECK
  cudaDeviceSynchronize();
  cudaCheckError();
#endif

#ifndef __PRINT__  
  printf("sm_count is %d\n",sm_count);
  printf("MAX shared memory is %f KB but only use %f KB\n",maxSharedMemory/1024.0,SharedMemoryUsed/1024.0);
  printf(" shared meomory size is %ld KB\n", executeSM/1024);

#endif

#ifdef __PRINT__
  #ifdef BASELINE
    #ifndef DA100X
      printf("bsln\t");
    #else
      printf("asyncbsln\t");
    #endif
  #endif 
  #ifdef BASELINE_CM
    #ifndef DA100X
      printf("bsln_cm\t");
    #else
      printf("asyncbsln_cm\t");
    #endif
  #endif 
  
  #ifdef NAIVE
    printf("naive\t");
  #endif 

  #ifdef PERSISTENT
    #ifndef DA100X
      printf("psstnt\t");
    #else
      printf("asyncpsstnt\t");
    #endif
  #endif

  // #ifdef GEN
  //     printf("gen"); 
  //   #else
  //     printf("asyncgen"); 
  //   #endif
  //   #if REG_FOLDER_Y==0 && SM_FOLER_Y ==0
  //     printf("\t");
  //   #endif
  //   #if REG_FOLDER_Y==0 && SM_FOLER_Y !=0
  //     printf("_sm\t");
  //   #endif
  //   #if REG_FOLDER_Y!=0 && SM_FOLER_Y ==0
  //     printf("_reg\t");
  //   #endif
  //   #if REG_FOLDER_Y!=0 && SM_FOLER_Y !=0
  //     printf("_mix\t");
  //   #endif
  // #endif
#endif 

#ifdef __PRINT__
  printf("%d\t%d\t",ptx,sizeof(REAL)/4);
  printf("%d\t%d\t%d\t",width_x,width_y,iteration);
  printf("<%d,%d>\t<%d,%d>\t%d\t0\t0\t",executeBlockDim.x,1,
        executeGridDim.x,executeGridDim.y,
        (executeGridDim.x)*(executeGridDim.y)/sm_count);
#endif

#ifdef _TIMER_
  cudaEventRecord(_forma_timer_stop_,0);
  cudaEventSynchronize(_forma_timer_stop_);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime,_forma_timer_start_,_forma_timer_stop_);
  #ifdef __PRINT__
    printf("%f\t%f\n",elapsedTime,(REAL)iteration*(width_y-2*halo)*(width_x-2*halo)/ elapsedTime/1000/1000);
  #else
    printf("[FORMA] Computation Time(ms) : %lf\n",elapsedTime);
    printf("[FORMA] Speed(GCells/s) : %lf\n",(REAL)iteration*(width_y)*(width_x)/ elapsedTime/1000/1000);
    printf("[FORMA] Speed(GFLOPS/s) : %lf\n", (REAL)17*iteration*(width_y)*(width_x)/ elapsedTime/1000/1000);
    printf("[FORMA] bandwidth(GB/s) : %lf\n", (REAL)sizeof(REAL)*iteration*((width_y)*(width_x)+width_x*width_y)/ elapsedTime/1000/1000);
    printf("[FORMA] width_x:width_y=%d:%d\n",(int)width_x, (int)width_y);
    printf("[FORMA] gdimx:gdimy=%d:%d\n",(int)executeGridDim.x, (int)executeGridDim.y);
    #if defined(GEN) || defined(MIX)
      printf("[FORMA] cached width_x:width_y=%d:%d\n",(int)TILE_X*grid_dim.x, (int)(max_sm_flder+REG_FOLDER_Y)*RTILE_Y*grid_dim.y);
      printf("[FORMA] cached b:sf:rf=%d:%d:%d\n", (int)RTILE_Y, (int)max_sm_flder, (int)REG_FOLDER_Y);
    #endif
  #endif

  cudaEventDestroy(_forma_timer_start_);
  cudaEventDestroy(_forma_timer_stop_);
#endif


//finalization
#ifdef CHECK
  // printf("check error here*\n");
  cudaDeviceSynchronize();
  cudaCheckError();
#endif

#if defined(GEN) || defined(PERSISTENT)
  if(iteration%2==1)
    cudaMemcpy(__var_0__,__var_2__, sizeof(REAL)*((width_y-0)*(width_x-0)), cudaMemcpyDeviceToHost);
  else
    cudaMemcpy(__var_0__,input, sizeof(REAL)*((width_y-0)*(width_x-0)), cudaMemcpyDeviceToHost);
#else
  cudaMemcpy(__var_0__,__var_2__, sizeof(REAL)*((width_y-0)*(width_x-0)), cudaMemcpyDeviceToHost);
#endif
/*Kernel Launch End */
/* Host Free Begin */
  cudaFree(input);
  cudaFree(__var_1__);
  cudaFree(__var_2__);

  // cudaFree(L2_cache);
  // cudaFree(L2_cache1);
  // cudaFree(L2_cache2);
#if defined(GEN) || defined(PERSISTENT)
  cudaFree(L2_cache3);
#endif
  // cudaFree(L2_cache4);

}

PERKS_INITIALIZE_ALL_TYPE(PERKS_DECLARE_INITIONIZATION_ITERATIVE);


