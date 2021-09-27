#include "./common/common.hpp"
#include "./common/jacobi_reference.hpp"
#include "./common/jacobi_cuda.cuh"
#include "config.cuh"
#include <cassert>
#include <cstdio>

//#ifndef HALO 
//#define HALO 1
//#endif

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

//template<class REAL>
//void runTest()
//{
//
//}
//#ifndef HALO 
//  #define HALO 1
//#endif

int main(int argc, char const *argv[])
{
  int width_x; 
  int width_y;
  int iteration;
  width_x=width_y=4096;
  iteration=4;
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
  jacobi_gold_iterative((REAL*)input, width_y, width_x, (REAL*)output_gold,iteration);
  jacobi_iterative((REAL*)input, width_y, width_x, (REAL*)output,iteration);
#endif

  int halo=HALO*iteration;

  REAL error =
    checkError2D<REAL>
    (width_x, (REAL*)output, (REAL*) output_gold, halo, width_y-halo, halo, width_x-halo);
  printf("[Test] RMS Error : %e\n",error);
  if (error > TOLERANCE)
    return -1;
  delete[] input;
  delete[] output;
  delete[] output_gold;

  #undef REAL
  /* code */
  return 0;
}


/*
int main(int argc, char** argv) {
  int width_y, width_x;
  int iteration;
  if (argc >= 3) {
    width_y = atoi(argv[1]);
    width_x = atoi(argv[2]);

    if(argc>=4)
    {
      iteration=atoi(argv[3]);
    }  
    else
    {
      iteration=3;
    }
  }
  else {
    iteration=3;
    // width_x=256*8*2;
    // width_x=256;
    // width_x=512;
    // width_x=1024;
    // width_x = 2048;
#ifndef MIX
    width_x=4096;
    width_y = width_x;
    // width_y = 8*10*5;
#else
    iteration=3;
    #ifndef TILE_X
      #define TILE_X (256)
    #endif
    // #ifndef GDIM_X
    //   #define GDIM_X (16)
    // #endif
    // #ifndef GDIM_Y
    //   #define GDIM_Y (5)
    // #endif
    #ifndef SFOLDER_Y
      #define SFOLDER_Y (1)
    #endif
    #ifndef RFOLDER_Y
      #define RFOLDER_Y (8)
    #endif

    #ifndef RTILE_Y
      #define RTILE_Y (8)
    #endif
    #ifndef MULTIPLIER
    #define MULTIPLIER (1)
    #endif
    #ifdef DA100
      int sm_count=108;
    #else
      int sm_count=80;
    #endif
    width_x=4096;//TILE_X*GDIM_X;
    width_y=(RTILE_Y*(SFOLDER_Y+RFOLDER_Y))*sm_count*MULTIPLIER/(4096/TILE_X);//GDIM_Y*(RTILE_Y*(SFOLDER_Y+RFOLDER_Y));
    // printf("<%d,%d,%d,%d>",GDIM_Y,RTILE_Y,SFOLDER_Y,RFOLDER_Y);
#endif
  }
#ifndef __PRINT__
  printf("widthx=%d,widthy=%d,iteration=%d\n",width_x,width_y,iteration);
// #else
   
#endif

  REAL (*input)[width_x] = (REAL (*)[width_x])
    getRandom2DArray<REAL>(width_y, width_x);
  REAL (*output)[width_x] = (REAL (*)[width_x])
    getZero2DArray<REAL>(width_y, width_x);
  REAL (*output_gold)[width_x] = (REAL (*)[width_x])
    getZero2DArray<REAL>(width_y, width_x);

  // jacobi_iterative((REAL*)input, width_y/4, width_x, (REAL*)output,10000);
  // jacobi_iterative((REAL*)input, width_y, width_x, (REAL*)output,10000);
  // jacobi_iterative((REAL*)input, width_y, width_x, (REAL*)output,1000000);
#ifndef CHECK
  // jacobi_iterative((REAL*)input, width_y, width_x, (REAL*)output,1);
  jacobi_iterative((REAL*)input, width_y, width_x, (REAL*)output,iteration);
#endif
#ifdef CHECK
  jacobi_iterative((REAL*)input, width_y, width_x, (REAL*)output,iteration);

  // jacobi_gold((REAL*)input, width_y, width_x, (REAL*)output);

  jacobi_gold_iterative((REAL*)input, width_y, width_x, (REAL*)output_gold,iteration);
  // printdiff(output[0],output_gold[0],width_x,width_y);

#ifdef PRINT_OUTPUT
  printf("Output :\n");
  print2DArray<REAL>(width_x, (REAL*)output, 0, width_y-0, 0, width_x-0);
  printf("\nOutput Gold:\n");
  print2DArray<REAL>(width_x, (REAL*)output_gold, 0, width_y-0, 0, width_x-0);
#endif

  REAL error =
    checkError2D<REAL>
    (width_x, (REAL*)output, (REAL*) output_gold, 0, width_y-0, 0, width_x-0);
  printf("[Test] RMS Error : %e\n",error);
  if (error > TOLERANCE)
    return -1;
#endif
  delete[] input;
  delete[] output;
  delete[] output_gold;
}
*/