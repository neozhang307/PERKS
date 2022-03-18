#include "../common/common.hpp"
#include "../common/jacobi_reference.hpp"
#include "../common/types.hpp"

//#ifndef REAL
//#define REAL float
//#endif
template<class REAL>
static void print_matrix 
(int width_y, int width_x, REAL* l_output)
{
  REAL (*output)[width_x] = (REAL (*)[width_x])l_output;
  for( int i = 2; i < width_y-2; i++ )
    for( int j = 2; j < width_x-2; j++ )
      printf ("Output[%d][%d] = %.6f\n", i, j, output[i][j]);
}

template<class REAL>
static void jacobi_step
(const REAL* l_input, int width_y, int width_x, REAL* l_output, int step)
{
  const REAL (*input)[width_x] = (const REAL (*)[width_x])l_input;
  REAL (*output)[width_x] = (REAL (*)[width_x])l_output;
  for( int i = 4; i < width_y-4; i++ )
    for( int j = 4 ; j < width_x-4; j++ )
      output[i][j] = (
        1*input[i-4][j] + 
        5*input[i-3][j] + 
        7*input[i-2][j] + 
        5*input[i-1][j] + 
        2*input[i][j-4] + 3*input[i][j-3] + 9*input[i][j-2] + 12*input[i][j-1] + 
        15*input[i][j] + 
        12*input[i][j+1] + 9*input[i][j+2] + 3*input[i][j+3]+  3*input[i][j+4]+ 
        5*input[i+1][j] + 
        7*input[i+2][j]+ 
        5*input[i+3][j]+ 
        4*input[i+4][j]
        ) / 118;
}

template<class REAL>
void jacobi_gold
(REAL* l_input, int width_y, int width_x, REAL* l_output)
{
  REAL* temp = getZero2DArray<REAL>(width_y, width_x);
  //jacobi_step(l_input, width_y, width_x, temp, 0);
  //jacobi_step(temp, width_y, width_x, l_output, 1);
  //jacobi_step(l_input, width_y, width_x, temp, 2);
  //jacobi_step(temp, width_y, width_x, l_output, 3);

  jacobi_step(l_input, width_y, width_x, temp, 0);
  //print_matrix (width_y, width_x, temp);
  jacobi_step(temp, width_y, width_x, l_output, 1);
  memset(temp, 0, sizeof(REAL) * width_y * width_x);
  jacobi_step(l_output, width_y, width_x, temp, 2);
  memset(l_output, 0, sizeof(REAL) * width_y * width_x);
  jacobi_step(temp, width_y, width_x, l_output, 3);
}
PERKS_INITIALIZE_ALL_TYPE(PERKS_DECLARE_INITIONIZATION_REFERENCE);