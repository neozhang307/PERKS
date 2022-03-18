#include "../common/common.hpp"
#include "../common/types.hpp"
#include "../common/jacobi_reference.hpp"

// #ifndef REAL
// #define REAL float
// #endif

template<class REAL>
static void print_matrix
(int height, int width_y, int width_x, REAL* l_output)
{
  REAL (*output)[width_y][width_x] = (REAL (*)[width_y][width_x])l_output;
  for( int i = 0 ; i < height; i++ )
    for( int j = 0 ; j < width_y; j++ )
      for (int k = 0 ; k < width_x; k++ ) 
	printf ("Input[%d][%d][%d] = %.6f\n", i, j, k, output[i][j][k]);
}
template<class REAL>
static void j3d_step
(const REAL* l_input, int height, int width_y, int width_x, REAL* l_output, int step)
{
  const REAL (*input)[width_y][width_x] =
    (const REAL (*)[width_y][width_x])l_input;
  REAL (*output)[width_y][width_x] = (REAL (*)[width_y][width_x])l_output;

  for (int i = 2; i < height-2; i++)
    for (int j = 2; j < width_y-2; j++)
      for (int k = 2; k < width_x-2; k++) {
        output[i][j][k] =
          0.083f * input[i][j][k+2] + 0.083f * input[i][j][k+1] +
          0.083f * input[i][j][k-1] + 0.083f * input[i][j][k-2] +
          0.083f * input[i][j+2][k] + 0.083f * input[i][j+1][k] +
          0.083f * input[i][j-1][k] + 0.083f * input[i][j-2][k] +
          0.083f * input[i+2][j][k] + 0.083f * input[i+1][j][k] +
          0.083f * input[i-1][j][k] + 0.083f * input[i-2][j][k] -
          0.996f * input[i][j][k];
	//if (step == 3) printf ("Output[%d][%d][%d] = %.6f (%6f, %6f, %6f, %6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f)\n", i, j, k, output[i][j][k], input[i+2][j][k], input[i+1][j][k], input[i-1][j][k], input[i-2][j][k], input[i][j][k], input[i][j][k-2], input[i][j][k-1], input[i][j][k+1], input[i][j][k+2], input[i][j-2][k], input[i][j-1][k], input[i][j+1][k], input[i][j+2][k]);
      }
}


template<class REAL>
void j3d_gold
(REAL *l_input, int height, int width_y, int width_x, REAL* l_output)
{
  //print_matrix(height, width_y, width_x, l_input); 
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