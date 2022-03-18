
#include "../common/common.hpp"
#include "../common/types.hpp"
#include "../common/jacobi_reference.hpp"



//#ifndef REAL
//#define REAL float
//#endif

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
 
  for( int l_y = 0 ; l_y < width_y-0 ; l_y++ )
  {
    for( int l_x = 0 ; l_x < width_x-0 ; l_x++ )
    {
      int w0 = (l_x-1>0?l_x-1:0);//(l_x == 0)        ? 0 : 0 - 1;
      int w1 = (l_x-2>0?l_x-2:0);//(l_x == 0)        ? 0 : 0 - 1;
      int e0 = (l_x+1<width_x-1?l_x+1:width_x-1);//(l_x == width_x-1)     ? 0 : 0 + 1;
      int e1 = (l_x+2<width_x-1?l_x+2:width_x-1);//(l_x == width_x-1)     ? 0 : 0 + 1;
      int s0 = (l_y-1>0?l_y-1:0);//(l_y == 0)        ? 0 : 0 - 1;
      int s1 = (l_y-2>0?l_y-2:0);//(l_y == 0)        ? 0 : 0 - 1;
      int n0 = (l_y+1<width_y-1?l_y+1:width_y-1);;//(l_y == width_y-1)     ? 0 : 0 + 1;    
      int n1 = (l_y+2<width_y-1?l_y+2:width_y-1);;//(l_y == width_y-1)     ? 0 : 0 + 1;    

      output[l_y][l_x] = 
      ( 
        1*  input[s1][w1] + 2*  input[s1][w0] + 3*  input[s1][l_x] + 4*input[s1] [e0] + 5*input[s1] [e1] + 
        7*  input[s0][w1] + 7*  input[s0][w0] + 5*  input[s0][l_x] + 7*input[s0] [e0] + 6*input[s0] [e1] + 
        8* input[l_y][w1] + 12* input[l_y][w0] + 15.0* input[l_y][l_x] + 12* input[l_y][e0] + 12* input[l_y][e1] +
        9*  input[n0][w1] + 9*  input[n0][w0] + 5*  input[n0][l_x] + 7*  input[n0][e0] + 15*  input[n0] [e1] +
        10*  input[n1][w1] + 11*  input[n1][w0] + 12*  input[n1][l_x] + 13*  input[n1][e0] + 14*  input[n1] [e1]
        ) / 118;  
    }  
  } 
  //if (step == 1) print_matrix(width_y, width_x, l_output);
}

// extern "C" void jacobi_gold
// (REAL* l_input, int width_y, int width_x, REAL* l_output)
// {
//   REAL* temp = getZero2DArray<REAL>(width_y, width_x);
//   // jacobi_step(l_input, width_y, width_x, l_output, 0);
// //4
//   // jacobi_step(l_input, width_y, width_x, temp, 0);
//   // jacobi_step(temp, width_y, width_x, l_output, 1);
//   // memset(temp, 0, sizeof(REAL) * width_y * width_x);
//   // jacobi_step(l_output, width_y, width_x, temp, 2);
//   // memset(l_output, 0, sizeof(REAL) * width_y * width_x);
//   // jacobi_step(temp, width_y, width_x, l_output, 3);
// //3
//   jacobi_step(l_input, width_y, width_x, l_output, 0);
//   memset(temp, 0, sizeof(REAL) * width_y * width_x);
//   jacobi_step(l_output, width_y, width_x, temp, 1);
//   memset(l_output, 0, sizeof(REAL) * width_y * width_x);
//   jacobi_step(temp, width_y, width_x, l_output, 2);
// }

template<class REAL>
void jacobi_gold_iterative
(REAL* l_input, int width_y, int width_x, REAL* l_output, int iteration)
{

  REAL* temp = getZero2DArray<REAL>(width_y, width_x);
  if(iteration%2==1)
  {
    jacobi_step(l_input, width_y, width_x, l_output, 0);
    for(int i=1; i<iteration; i++)
    {
      jacobi_step( l_output, width_y, width_x, temp, i);
      REAL *temp2=temp;
      temp=l_output;
      l_output=temp2;
    }
  }
  else
  {
    jacobi_step(l_input, width_y, width_x, temp, 0);
    for(int i=1; i<iteration; i++)
    {
      jacobi_step(temp, width_y, width_x, l_output, i);
      REAL *temp2=temp;
      temp=l_output;
      l_output=temp2;
    }
  }
}

PERKS_INITIALIZE_ALL_TYPE(PERKS_DECLARE_INITIONIZATION_REFERENCE_ITERATIVE);
