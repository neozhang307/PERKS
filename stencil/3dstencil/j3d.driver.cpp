#include "./common/common.hpp"
#include "./common/jacobi_reference.hpp"
#include "./common/jacobi_cuda.cuh"
#include "./config.cuh"
#include <cassert>
#include <cstdio>

#ifndef REAL
#define REAL float
#endif

#define TOLERANCE 1e-5

int main(int argc, char** argv) {
  // int sm_count;
  // cudaDeviceGetAttribute ( &sm_count, cudaDevAttrMultiProcessorCount,0 );
  int height, width_y, width_x;
  int iteration=3;
  if (argc >= 4) {
    height = atoi(argv[1]);
    width_y = atoi(argv[2]);
    width_x = atoi(argv[3]);
    if(argc>=5)
    {
      iteration=atoi(argv[4]);
    }  
    else
    {
      iteration=3;
    }
  }
  else {
    height = 256;
    width_y = 216;//128;
    // width_y = 60;
    width_x = 256;
  }

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
  j3d_iterative((REAL*)input, height, width_y, width_x, (REAL*)output,iteration);
  j3d_gold_iterative((REAL*)input, height, width_y, width_x, (REAL*)output_gold,iteration);

#endif

  int domain_hallo=0;//HALO*2;
  REAL error =
    checkError3D<REAL>
    (width_y, width_x, (REAL*)output, (REAL*) output_gold, domain_hallo, height-domain_hallo, domain_hallo,
     width_y-domain_hallo, domain_hallo, width_x-domain_hallo);
  printf("[Test] RMS Error : %e\n",error);
  if (error > TOLERANCE)
    return -1;

  delete[] input;
  delete[] output;
  delete[] output_gold;
}
