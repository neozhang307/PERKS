#include "../common/common.hpp"
#include "../common/types.hpp"
#include "../common/jacobi_reference.hpp"



template<class REAL>
void j3d_step
(const REAL* l_input, int height, int width_y, int width_x, REAL* l_output, int step)
{
  const REAL (*input)[width_y][width_x] =
    (const REAL (*)[width_y][width_x])l_input;
  REAL (*output)[width_y][width_x] = (REAL (*)[width_y][width_x])l_output;

  for (int i = 1; i < height-1; i++)
    for (int j = 1; j < width_y-1; j++)
      for (int k = 1; k < width_x-1; k++) {
        output[i][j][k] =
          0.161f * input[i][j][k+1] + 0.162f * input[i][j][k-1] +
          0.163f * input[i][j+1][k] + 0.164f * input[i][j-1][k] +
          0.165f * input[i+1][j][k] + 0.166f * input[i-1][j][k] -
          1.67f * input[i][j][k];
	  //if (step == 0) printf ("output[%d][%d][%d] = %.6f (%.6f)\n", i, j, k, output[i][j][k], input[i+1][j][k]);
      }
}


template<class REAL>
void j3d_gold
(REAL *l_input, int height, int width_y, int width_x, REAL* l_output)
{
//  const REAL (*input)[width_y][width_x] = (const REAL (*)[width_y][width_x])l_input;
//  for (int i = 0; i < height; i++) 
//    for (int j = 0; j< width_y; j++) 
//	for (int k = 0; k < width_x; k++)  
//	  printf ("input[%d][%d][%d] = %.6f\n", i, j, k, input[i][j][k]);
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