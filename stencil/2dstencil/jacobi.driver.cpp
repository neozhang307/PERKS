#include "./common/common.hpp"
// #include "./common/cuda_common.cuh"
// #include <cuda_runtime.h>
#include "./common/jacobi_reference.hpp"
#include "./common/jacobi_cuda.cuh"
#include "config.cuh"
#include <cassert>
#include <cstdio>

#define TOLERANCE 1e-5

template<class REAL>
void printdiff(REAL* out, REAL* ref, int widthx, int widthy)
{
  for(int y=0; y<widthy; y++)
  {
    for(int x=0; x<widthx; x++)
    {
      REAL err = out[y*widthx+x]-ref[y*widthx+x];
      err=err<0?-err:err;
      if(err>0.00001)
      {
        printf("(%d:%d)=%f:%f ", x,y,out[y*widthx+x],ref[y*widthx+x]);
      }
    }
  }
}

#include "./common/cub_utils.cuh"

int main(int argc, char  *argv[])
{
  int width_x; 
  int width_y;
  int iteration=3;
  width_x=width_y=4096;//4096;
  bool fp32=true;//float
  bool check=false;
  int bdimx=256;
  bool async=false;


  if (argc >= 3) {
    width_y = atoi(argv[1]);
    width_x = atoi(argv[2]);
    width_x = width_x==0?2048:width_x;
    width_y = width_y==0?2048:width_y;
    // if(argc>=4)
    // {
    //   iteration=atoi(argv[3]);
    // }
  }

  CommandLineArgs args(argc, argv);
  fp32 = args.CheckCmdLineFlag("fp32");
  async = args.CheckCmdLineFlag("async");
  check = args.CheckCmdLineFlag("check");
  // bdimx = args
  args.GetCmdLineArgument("bdim", bdimx);
  args.GetCmdLineArgument("iter", iteration);
  if(bdimx==0)bdimx=256;
  if(iteration==0)iteration=3;

#ifdef REFCHECK
  iteration=4;
#endif

  if(fp32)
  {
    #define REAL float
    
    REAL (*input)[width_x] = (REAL (*)[width_x])
      getRandom2DArray<REAL>(width_y, width_x);
    REAL (*output)[width_x] = (REAL (*)[width_x])
      getZero2DArray<REAL>(width_y, width_x);
    REAL (*output_gold)[width_x] = (REAL (*)[width_x])
      getZero2DArray<REAL>(width_y, width_x);
    #ifdef REFCHECK
      jacobi_gold((REAL*)input, width_y, width_x, (REAL*)output);
      jacobi_gold_iterative((REAL*)input, width_y, width_x, (REAL*)output_gold,iteration);
    #else
      jacobi_iterative((REAL*)input, width_y, width_x, (REAL*)output,bdimx,iteration,async);
      if(check!=0)
      {
        jacobi_gold_iterative((REAL*)input, width_y, width_x, (REAL*)output_gold,iteration);
      }
    #endif

    
    if(check!=0){
      int halo=HALO*iteration;
      double error =
        checkError2D<REAL>
        (width_x, (REAL*)output, (REAL*) output_gold, halo, width_y-halo, halo, width_x-halo);
      printf("[Test] RMS Error : %e\n",error);
      if (error > TOLERANCE)
        return -1;
    }
    #undef REAL
    delete[] input;
    delete[] output;
    delete[] output_gold;
  }
  else
  {
    #define REAL double
  
    REAL (*input)[width_x] = (REAL (*)[width_x])
      getRandom2DArray<REAL>(width_y, width_x);
    REAL (*output)[width_x] = (REAL (*)[width_x])
      getZero2DArray<REAL>(width_y, width_x);
    REAL (*output_gold)[width_x] = (REAL (*)[width_x])
      getZero2DArray<REAL>(width_y, width_x);
    #ifdef REFCHECK
      jacobi_gold((REAL*)input, width_y, width_x, (REAL*)output);
      jacobi_gold_iterative((REAL*)input, width_y, width_x, (REAL*)output_gold,iteration);
    #else
      jacobi_iterative((REAL*)input, width_y, width_x, (REAL*)output, bdimx, iteration, async);
      
  
      if(check!=0)
      {
        jacobi_gold_iterative((REAL*)input, width_y, width_x, (REAL*)output_gold,iteration);
      }
    #endif

    #ifdef REFCHECK
      int halo=HALO*iteration;
    #else
      int halo=HALO*iteration;
    #endif
    if(check!=0){
      
      double error =
        checkError2D<REAL>
        (width_x, (REAL*)output, (REAL*) output_gold, halo, width_y-halo, halo, width_x-halo);
      
      printf("[Test] RMS Error : %e\n",error);
      if (error > TOLERANCE)
        return -1;
    }
    #undef REAL
    delete[] input;
    delete[] output;
    delete[] output_gold;
  }
  // Check_CUDA_Error("147");

// #define cudaCheckError() {                                          \
//  cudaError_t e=cudaGetLastError();                                 \
//  if(e!=cudaSuccess) {                                              \
//    printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
//  }                                                                 \
// }
  // cudaDeviceSynchronize();
  // cudaError_t e=cudaGetLastError();                                 
  //  if(e!=cudaSuccess) {                                              
  //    printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           
  //  } 
  // #undef REAL
  /* code */
  return 0;
}
