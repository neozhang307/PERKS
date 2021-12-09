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

  for (int i = 1; i < height-1; i++)
    for (int j = 1; j < width_y-1; j++)
      for (int k = 1; k < width_x-1; k++) {
        output[i][j][k] = (0.5*input[i-1][j-1][k-1] + 0.7*input[i-1][j-1][k] + 0.9*input[i-1][j-1][k+1] + 1.2*input[i-1][j][k-1] + 1.5*input[i-1][j][k] + 1.2*input[i-1][j][k+1] + 0.9*input[i-1][j+1][k-1] + 0.7*input[i-1][j+1][k] + 0.5*input[i-1][j+1][k+1] +
                           0.51*input[i][j-1][k-1] + 0.71*input[i][j-1][k] + 0.91*input[i][j-1][k+1] + 1.21*input[i][j][k-1] + 1.51*input[i][j][k] + 1.21*input[i][j][k+1] + 0.91*input[i][j+1][k-1] + 0.71*input[i][j+1][k] + 0.51*input[i][j+1][k+1] +
                           0.52*input[i+1][j-1][k-1] + 0.72*input[i+1][j-1][k] + 0.92*input[i+1][j-1][k+1] + 1.22*input[i+1][j][k-1] + 1.52*input[i+1][j][k] + 1.22*input[i+1][j][k+1] + 0.92*input[i+1][j+1][k-1] + 0.72*input[i+1][j+1][k] + 0.52*input[i+1][j+1][k+1]) / 159;
      }
}


// extern "C"
template<class REAL> 
void j3d_gold
(REAL *l_input, int height, int width_y, int width_x, REAL* l_output)
{
  REAL* temp = getZero3DArray<REAL>(height, width_y, width_x);
  j3d_step(l_input, height, width_y, width_x, temp, 0);
  j3d_step(temp, height, width_y, width_x, l_output, 1);
  memset(temp, 0, sizeof(REAL) * height * width_y * width_x);
  j3d_step(l_output, height, width_y, width_x, temp, 2);
  memset(l_output, 0, sizeof(REAL) * height * width_y * width_x);
  j3d_step(temp, height, width_y, width_x, l_output, 3);
  delete[] temp;
}


PERKS_INITIALIZE_ALL_TYPE(PERKS_DECLARE_INITIONIZATION_REFERENCE);