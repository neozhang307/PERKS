#include "../common/common.hpp"
#include "../common/types.hpp"
#include "../common/jacobi_reference.hpp"

// #ifndef REAL
// #define REAL float
// #endif
template<class REAL>
static void j3d_step
(const REAL* l_input, int height, int width_y, int width_x, REAL* l_output, int step)
{
  const REAL (*input)[width_y][width_x] =
    (const REAL (*)[width_y][width_x])l_input;
  REAL (*output)[width_y][width_x] = (REAL (*)[width_y][width_x])l_output;

  for (int l_h = 0; l_h < height-0; l_h++)
    for (int l_y = 0; l_y < width_y-0; l_y++)
      for (int l_x = 0; l_x < width_x-0; l_x++) {
        int b0=(l_h-1>0?l_h-1:0);
        int t0=(l_h+1<=height-1?l_h+1:height-1);
        int w0 = (l_x-1>0?l_x-1:0);
        int e0 = (l_x+1<width_x-1?l_x+1:width_x-1);
        int n0 = (l_y+1<width_y-1?l_y+1:width_y-1);
        int s0 = (l_y-1>0?l_y-1:0);

        output[l_h][l_y][l_x] = 
        // 1.51*input[l_h][l_y][l_x]/159;
            (
                0.5*input[b0][s0][w0]  + 0.7*input[b0][s0][l_x]  + 0.9*input[b0][s0][e0] + 
                1.2*input[b0][l_y][w0] + 1.5*input[b0][l_y][l_x] + 1.2*input[b0][l_y][e0] + 
                0.9*input[b0][n0][w0]  + 0.7*input[b0][n0][l_x]  + 0.5*input[b0][n0][e0] +

                0.51*input[l_h][s0][w0] + 0.71*input[l_h][s0][l_x] + 0.91*input[l_h][s0][e0] + 
                1.21*input[l_h][l_y][w0] + 1.51*input[l_h][l_y][l_x] + 1.21*input[l_h][l_y][e0] + 
                0.91*input[l_h][n0][w0] + 0.71*input[l_h][n0][l_x] + 0.51*input[l_h][n0][e0] +
                
                0.52*input[t0][s0][w0]  + 0.72*input[t0][s0][l_x]  + 0.92*input[t0][s0][e0] + 
                1.22*input[t0][l_y][w0] + 1.52*input[t0][l_y][l_x] + 1.22*input[t0][l_y][e0] + 
                0.92*input[t0][n0][w0]  + 0.72*input[t0][n0][l_x]  + 0.52*input[t0][n0][e0] 
                ) / 159;
      }
}

// extern "C" 
template<class REAL>
void j3d_gold_iterative
(REAL *l_input, int height, int width_y, int width_x, REAL* l_output,int iteration)
{
  REAL* temp = getZero3DArray<REAL>(height, width_y, width_x);
  // j3d_step(l_input, height, width_y, width_x, temp, 0);
  // j3d_step(temp, height, width_y, width_x, l_output, 1);
  // memset(temp, 0, sizeof(REAL) * height * width_y * width_x);
  // j3d_step(l_output, height, width_y, width_x, temp, 2);
  // memset(l_output, 0, sizeof(REAL) * height * width_y * width_x);
  // j3d_step(temp, height, width_y, width_x, l_output, 3);
  
  if(iteration%2==1)
  {
    // jacobi_step(l_input, width_y, width_x, l_output, 0);
    j3d_step(l_input, height, width_y, width_x, l_output, 0);
    // j3d_step(l_input, height, width_y, width_x, temp, 0);
    for(int i=1; i<iteration; i++)
    {
      // j3d_step(l_input, height, width_y, width_x, temp, i);
      j3d_step( l_output,height, width_y, width_x, temp, i);
      REAL *temp2=temp;
      temp=l_output;
      l_output=temp2;
    }
  }
  else
  {
    // jacobi_step(l_input, width_y, width_x, temp, 0);
    j3d_step(l_input, height, width_y, width_x, temp, 0);
    for(int i=1; i<iteration; i++)
    {
      // jacobi_step(temp, width_y, width_x, l_output, i);
      j3d_step(temp, height, width_y, width_x, l_output, i);
      REAL *temp2=temp;
      temp=l_output;
      l_output=temp2;
    }
  }

  // delete[] temp;
}

PERKS_INITIALIZE_ALL_TYPE(PERKS_DECLARE_INITIONIZATION_REFERENCE_ITERATIVE);