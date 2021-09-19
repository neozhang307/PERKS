//#include <sys/time.h>
#include "../common/common.hpp"
#include "../common/types.hpp"
#include "../common/jacobi_reference.hpp"

//#ifndef REAL
//#define REAL float
//#endif


//double rtclock () {
//	struct timezone Tzp;
//	struct timeval Tp;
//	int stat = gettimeofday (&Tp, &Tzp);
//	if (stat != 0) 
//		printf ("Error return from gettimeofday: %d", stat);
//	return (Tp.tv_sec + Tp.tv_usec*1.0e-6);
//}  

template<class REAL>
static void print_matrix 
(int width_y, int width_x, REAL* l_output)
{
  REAL (*output)[width_x] = (REAL (*)[width_x])l_output;
  for( int i = 1 ; i < width_y-1 ; i++ )
    for( int j = 1 ; j < width_x-1 ; j++ )
      printf ("Output[%d][%d] = %.6f\n", i, j, output[i][j]);
}

template<class REAL>
static void jacobi_step
(const REAL* l_input, int width_y, int width_x, REAL* l_output, int step)
{
  const REAL (*input)[width_x] = (const REAL (*)[width_x])l_input;
  REAL (*output)[width_x] = (REAL (*)[width_x])l_output;
  for( int i = 2 ; i < width_y-2 ; i++ )
    for( int j = 2 ; j < width_x-2 ; j++ )
      output[i][j] = (
        1*input[i-2][j-2] + 2*input[i-2][j-1] + 3*input[i-2][j] + 4*input[i-2][j+1] + 5*input[i-2][j+2] +
        7*input[i-1][j-2] + 7*input[i-1][j-1] + 5*input[i-1][j] + 7*input[i-1][j+1] + 6*input[i-1][j+2] +
        8*input[i][j-2] + 12*input[i][j-1] + 15*input[i][j] + 12*input[i][j+1] +  12*input[i][j+2] + 
        9*input[i+1][j-2] + 9*input[i+1][j-1] + 5*input[i+1][j] + 7*input[i+1][j+1] + 15*input[i+1][j+2] +
        10*input[i+2][j-2] + 11*input[i+2][j-1] + 12*input[i+2][j] + 13*input[i+2][j+1] + 14*input[i+2][j+2]
        ) 
    / 118;
  //if (step == 1) print_matrix(width_y, width_x, l_output);
}

template<class REAL>
void jacobi_gold
(REAL* l_input, int width_y, int width_x, REAL* l_output)
{
  REAL* temp = getZero2DArray<REAL>(width_y, width_x);
  //REAL start_time = rtclock ();
  jacobi_step(l_input, width_y, width_x, temp, 0);
//  REAL end_time = rtclock ();
  //printf ("Time taken by reference : %lf secs\n", end_time - start_time);
  //print_matrix (width_y, width_x, temp);
  jacobi_step(temp, width_y, width_x, l_output, 1);
  memset(temp, 0, sizeof(REAL) * width_y * width_x);
  jacobi_step(l_output, width_y, width_x, temp, 2);
  memset(l_output, 0, sizeof(REAL) * width_y * width_x);
  jacobi_step(temp, width_y, width_x, l_output, 3);
}

PERKS_INITIALIZE_ALL_TYPE(PERKS_DECLARE_INITIONIZATION_REFERENCE);