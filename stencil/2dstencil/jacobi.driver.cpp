#include "./common/common.hpp"
#include "./common/jacobi_reference.hpp"
#include "./common/jacobi_cuda.cuh"
#include "config.cuh"
#include <cassert>
#include <cstdio>

#define TOLERANCE 1e-5
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

int main(int argc, char const *argv[])
{
  int width_x; 
  int width_y;
  int iteration=3;
  width_x=width_y=4096;//4096;
  int type=0;//float
  int check=0;
  bool async=false;
  // width_y=100;

  if (argc >= 3) {
    width_y = atoi(argv[1]);
    width_x = atoi(argv[2]);

    if(argc>=4)
    {
      iteration=atoi(argv[3]);
      if(argc>=5)
      {
        type=atoi(argv[4])!=0?1:0;
        if(argc>=6)
        {
          // async=atoi(argv[4])!=0?true:false;
          // if(argc>=7)
          // {
            check=atoi(argv[5])!=0?1:0;
          // }
        }
      }
    }  
    else
    {
      iteration=3;
    }
  }
#ifdef REFCHECK
  iteration=4;
#endif
// #endif

  if(type==0)
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
      if(check!=0)
      {
        jacobi_gold_iterative((REAL*)input, width_y, width_x, (REAL*)output_gold,iteration);
      }
      jacobi_iterative((REAL*)input, width_y, width_x, (REAL*)output,iteration);
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
      if(check!=0)
      {
        jacobi_gold_iterative((REAL*)input, width_y, width_x, (REAL*)output_gold,iteration);
      }
      jacobi_iterative((REAL*)input, width_y, width_x, (REAL*)output,iteration);

    #endif

    #ifdef REFCHECK
      int halo=HALO*iteration;
    #else
      int halo=HALO*iteration;
      // int halo=0;
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


  // #undef REAL
  /* code */
  return 0;
}
