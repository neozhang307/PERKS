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

#ifdef REFCHECK
#define CPUCODE
#endif


#include "./common/cub_utils.cuh"

int main(int argc, char  *argv[])
{
  int width_x; 
  int width_y;
  int iteration=3;
  int warmupiteration=-1;
  width_x=width_y=2048;//4096;
  bool fp32=true;//float
  bool check=false;
  int bdimx=256;
  int blkpsm=0;

  bool async=false;
  bool useSM=false;
  bool usewarmup=false;
  bool checkmindomain=false;
  bool usesmall=false;

  if (argc >= 3) {
    width_y = atoi(argv[1]);
    width_x = atoi(argv[2]);
    width_x = width_x==0?2048:width_x;
    width_y = width_y==0?2048:width_y;
  }

  CommandLineArgs args(argc, argv);
  checkmindomain = args.CheckCmdLineFlag("checkmindomain");
  fp32 = args.CheckCmdLineFlag("fp32");
  async = args.CheckCmdLineFlag("async");
  check = args.CheckCmdLineFlag("check");
  useSM = args.CheckCmdLineFlag("usesm");
  usewarmup = args.CheckCmdLineFlag("warmup");
  usesmall = args.CheckCmdLineFlag("small");
  // bdimx = args
  args.GetCmdLineArgument("bdim", bdimx);
  args.GetCmdLineArgument("iter", iteration);
  args.GetCmdLineArgument("warmiter", warmupiteration);
  args.GetCmdLineArgument("blkpsm", blkpsm);
  if(bdimx==0)bdimx=256;
  if(iteration==0)iteration=3;

#ifndef CPUCODE
  if(usesmall)
  {
    int registers=256;
    if(blkpsm==2)registers=128;
    if(blkpsm==1)registers=256;
    // printf("hrere");
    if(fp32)
    {
      width_y=getMinWidthY<float>(width_x,bdimx,registers,useSM,blkpsm);
    }
    else 
    {
      width_y=getMinWidthY<double>(width_x,bdimx,registers,useSM,blkpsm);
    }
    if(width_y==0)
    {
      printf("error unsupport no cache version small code\n");
      return 0;
    }
    // printf("widis %d\n",width_y);
  }

  if(checkmindomain)
  {
    if(fp32)
    {
      printf("2048 %d\n",getMinWidthY<float>(2048,bdimx));
    }
    else
    {
      printf("2048 %d\n",getMinWidthY<double>(2048,bdimx));
    }
    return 0;
  }


#ifndef __PRINT__
  {
  int registers=256;
  if(blkpsm==2)registers=128;
  if(blkpsm==1)registers=256;
  // #ifdef GEN
  // registers=0;
  // printf("0\n");
  // #endif
  if(fp32)
  {
    printf("%d %d\n", width_x,getMinWidthY<float>(width_x,bdimx,registers,useSM,blkpsm));
  }
  else
  {
    printf("%d %d\n", width_x,getMinWidthY<double>(width_x,bdimx,registers,useSM,blkpsm));
  }
}
#endif
#endif
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
      jacobi_iterative((REAL*)input, width_y, width_x, (REAL*)output,bdimx,blkpsm,iteration,async,useSM,usewarmup, warmupiteration);
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
      jacobi_iterative((REAL*)input, width_y, width_x, (REAL*)output, bdimx, blkpsm, iteration, async, useSM,usewarmup, warmupiteration);
      
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

  return 0;
}
