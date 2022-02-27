#include "./common/common.hpp"
#include "./common/jacobi_reference.hpp"
#include "./common/jacobi_cuda.cuh"
#include "./config.cuh"
#include <cassert>
#include <cstdio>

// #ifndef REAL
// #define REAL float
// #endif

#define TOLERANCE 1e-5
#include "./common/cub_utils.cuh"

int main(int argc, char** argv) {
  // int sm_count;
  // cudaDeviceGetAttribute ( &sm_count, cudaDevAttrMultiProcessorCount,0 );
  int height, width_y, width_x;
  height = width_y = width_x = 256;

  bool fp32=true;//float
  bool check=false;
  int bdimx=256;//////////////might be a issue?
  int blkpsm=0;

  bool useSM=false;
  bool usewarmup=false;
  bool checkmindomain=false;
  bool usesmall=false;
  bool isDoubleTile=false;

  int iteration=3;
  int warmupiteration=-1;
  if (argc >= 3) {
    height = atoi(argv[1]);
    width_y = atoi(argv[2]);
    width_x = atoi(argv[3]);

    height = height<=0?256:height;
    width_x = width_x<=0?256:width_x;
    width_y = width_y<=0?256:width_y;
  }


  CommandLineArgs args(argc, argv);
  checkmindomain = args.CheckCmdLineFlag("checkmindomain");
  fp32 = args.CheckCmdLineFlag("fp32");
  check = args.CheckCmdLineFlag("check");
  useSM = args.CheckCmdLineFlag("usesm");
  usewarmup = args.CheckCmdLineFlag("warmup");
  usesmall = args.CheckCmdLineFlag("small");
  isDoubleTile = args.CheckCmdLineFlag("doubletile");
  // bdimx = args
  args.GetCmdLineArgument("bdim", bdimx);/////////////////////might be a issue
  args.GetCmdLineArgument("iter", iteration);
  args.GetCmdLineArgument("warmiter", warmupiteration);
  args.GetCmdLineArgument("blkpsm", blkpsm);

  if(bdimx==0)bdimx=256;////////////////////////might be a issue
  if(iteration==0)iteration=3;
  #ifndef REFCHECK
    if(usesmall)
    {
      if(fp32)
      {
        height=
              j3d_iterative<float>(nullptr,
                              height, width_y, width_x,
                              nullptr, 
                              bdimx, 
                              blkpsm, 
                              1, 
                              useSM,
                              false, 
                              0,
                              true);

      }
      else 
      {
        height=
              j3d_iterative<double>(nullptr,
                              height, width_y, width_x,
                              nullptr, 
                              bdimx, 
                              blkpsm, 
                              1, 
                              useSM,
                              false, 
                              0,
                              true);
      }

      if(height==0)
      {
        if(check)
        {
          printf("error unsupport no cache version small code\n");
        }
        return 0;
      }
      // printf("widis %d\n",width_y);
    }

    if(checkmindomain)
    {
      if(fp32)
      {
        printf("%d %d %d\n",getMinWidthY<float>(width_x, width_y, bdimx),width_y,width_x);
      }
      else
      {
        printf("%d %d %d\n",getMinWidthY<double>(width_x, width_y, bdimx),width_y,width_x);
      }
      return 0;
    }
  #endif

  if(fp32)
  {
    // printf("asdfsadfs");
    #define REAL float
      REAL (*input)[width_y][width_x] = (REAL (*)[width_y][width_x])
        getRandom3DArray<REAL>(height, width_y, width_x);
      REAL (*output)[width_y][width_x] = (REAL (*)[width_y][width_x])
        getZero3DArray<REAL>(height, width_y, width_x);
      REAL (*output_gold)[width_y][width_x] = (REAL (*)[width_y][width_x])
        getZero3DArray<REAL>(height, width_y, width_x);

      iteration=iteration==0?3:iteration;
    #ifdef REFCHECK
      iteration=8;
      // j3d_iterative((REAL*)input, height, width_y, width_x, (REAL*)output,iteration);
      j3d_gold((REAL*)input, height, width_y, width_x, (REAL*)output);
      j3d_gold_iterative((REAL*)input, height, width_y, width_x, (REAL*)output_gold,iteration);
    #else
      int err = j3d_iterative((REAL*)input, height, width_y, width_x, (REAL*)output, bdimx, blkpsm, iteration, useSM,usewarmup, warmupiteration);
      if(err==-1)
      {
        printf("unsupport setting, no free space for cache with shared memory\n");
        check=0;
      }
      if(check!=0)
      {
        j3d_gold_iterative((REAL*)input, height, width_y, width_x, (REAL*)output_gold,iteration);  
      }

    #endif
      if(check!=0){
        int domain_hallo=0;//HALO*2;
        REAL error =
          checkError3D<REAL>
          (width_y, width_x, (REAL*)output, (REAL*) output_gold, domain_hallo, height-domain_hallo, domain_hallo,
           width_y-domain_hallo, domain_hallo, width_x-domain_hallo);
        printf("[Test] RMS Error : %e\n",error);
        if (error > TOLERANCE)
          return -1;
      }
      delete[] input;
      delete[] output;
      delete[] output_gold;
      #undef REAL
  }
  else
  {
    #define REAL double
      REAL (*input)[width_y][width_x] = (REAL (*)[width_y][width_x])
        getRandom3DArray<REAL>(height, width_y, width_x);
      REAL (*output)[width_y][width_x] = (REAL (*)[width_y][width_x])
        getZero3DArray<REAL>(height, width_y, width_x);
      REAL (*output_gold)[width_y][width_x] = (REAL (*)[width_y][width_x])
        getZero3DArray<REAL>(height, width_y, width_x);

      iteration=iteration==0?3:iteration;
    #ifdef REFCHECK
      iteration=8;
      // j3d_iterative((REAL*)input, height, width_y, width_x, (REAL*)output,iteration);
      j3d_gold((REAL*)input, height, width_y, width_x, (REAL*)output);
      j3d_gold_iterative((REAL*)input, height, width_y, width_x, (REAL*)output_gold,iteration);
    #else
      int err = j3d_iterative((REAL*)input, height, width_y, width_x, (REAL*)output, bdimx, blkpsm, iteration, useSM,usewarmup, warmupiteration);
      // printf("err is %d\n",err);
      if(err==-1)
      {
        if(check)printf("unsupport setting, no free space for cache with shared memory\n");
        check=0;
      }
      if(check!=0)
      {
        j3d_gold_iterative((REAL*)input, height, width_y, width_x, (REAL*)output_gold,iteration);  
      }

    #endif
      if(check!=0){
        int domain_hallo=0;//HALO*2;
        REAL error =
          checkError3D<REAL>
          (width_y, width_x, (REAL*)output, (REAL*) output_gold, domain_hallo, height-domain_hallo, domain_hallo,
           width_y-domain_hallo, domain_hallo, width_x-domain_hallo);
        printf("[Test] RMS Error : %e\n",error);
        if (error > TOLERANCE)
          return -1;
      }
      delete[] input;
      delete[] output;
      delete[] output_gold;
      #undef REAL
  }

}
