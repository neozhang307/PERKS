
__device__ void gpuSpMV_perk(int *I, int *J, float *val, int nnz, int num_rows,
                        float alpha, float *inputVecX, float *outputVecY,
                        cg::thread_block &cta, const cg::grid_group &grid) {
  for (int i = grid.thread_rank(); i < num_rows; i += grid.size()) {
    int row_elem = I[i];
    int next_row_elem = I[i + 1];
    int num_elems_this_row = next_row_elem - row_elem;

    float output = 0.0;
    for (int j = 0; j < num_elems_this_row; j++) {
      // I or J or val arrays - can be put in shared memory
      // as the access is random and reused in next calls of gpuSpMV function.
      output += alpha * val[row_elem + j] * inputVecX[J[row_elem + j]];
    }

    outputVecY[i] = output;
  }
}
template<bool pre, bool post>
void __device__ gpuCopyDiaMatrToSM_perk(float* sm_mat, int sm_start, int sm_step,
                                    float*origin, int g_start ,int g_step, 
                                    int size)
{
  for (int i = g_start, i_mat=sm_start; 
    i < size; 
    i += g_step, i_mat+=sm_step) {  
    _Pragma("unroll")
    for (int j = -1; j < 2; j++) {
      if(pre && i+j<0)continue;
      if(post && i+j>=size)continue;
      (sm_mat)[i_mat*3+j] = (origin)[3*i + j] ;
    }
  }
}

#define gpuCopyDiaMatr_perk_reg(x,x_start,x_step,y,y_start,y_step,size,pre,post) \
do{\
  _Pragma("unroll")\
  for(int i=0; i<size; i++)\
  {\
    int y_ind= y_start+i*y_step;\
    int x_ind= x_start+i*x_step;\
    y[y_ind*3-1] = (pre&&i==0)?0: x[x_ind*3-1];\
    y[y_ind*3] = x[x_ind*3];\
    y[y_ind*3+1] = (post&&i==size-1)?0:x[x_ind*3+1];\
  }\
}while(0)


template <typename        ValueT,
          typename        OffsetT>
__device__ void gpuSaxpy_perk(ValueT *x, OffsetT x_start, OffsetT x_stop, OffsetT x_step,
                              ValueT *y, OffsetT y_start, OffsetT y_stop, OffsetT y_step,
                              ValueT a) {
  _Pragma("unroll")
  for(int x_ind=x_start, y_ind=y_start; 
    x_ind<x_stop&&y_ind<y_stop; 
    x_ind+=x_step, y_ind+=y_step)
  {
    y[y_ind] = a * x[x_ind] + y[y_ind];
  }
}

template <typename        ValueT,
          typename        OffsetT>
__device__ void gpuSaxpy_perk(ValueT *x, 
                              ValueT *y, OffsetT start, OffsetT stop, OffsetT step,
                              ValueT a) {
  _Pragma("unroll")
  for(int ind=start; ind<stop; ind+=step)
  {
    y[ind] = a * x[ind] + y[ind];
  }
}




#define gpuSaxpy_perk_reg(x,x_start,x_step,y,y_start,y_step,size,a) \
do{\
  _Pragma("unroll")\
  for(int i=0; i<size; i++)\
  {\
    int y_ind= y_start+i*y_step;\
    int x_ind= x_start+i*x_step;\
    y[y_ind] = a * x[x_ind] + y[y_ind];\
  }\
}while(0)

template <typename        ValueT,
          typename        OffsetT>
__device__ void gpuDotProduct_perk_pre( ValueT *vecA, OffsetT A_start, OffsetT A_stop, OffsetT A_step,
                                        ValueT *vecB, OffsetT B_start, OffsetT B_stop, OffsetT B_step,
                              double &temp_sum) {
  _Pragma("unroll")
  for(int A_ind=A_start, B_ind=B_start; 
    A_ind<A_stop&&B_ind<B_stop; 
    A_ind+=A_step, B_ind+=B_step)
  {
    temp_sum += static_cast<double>(vecA[A_ind] * vecB[B_ind]);
  }
}
template <typename        ValueT,
          typename        OffsetT>
__device__ void gpuDotProduct_perk_pre( ValueT *vecA, 
                                        ValueT *vecB, OffsetT start, OffsetT stop, OffsetT step,
                              double &temp_sum) {
  _Pragma("unroll")
  for(int ind=start; ind<stop; ind+=step)
  {
    temp_sum += static_cast<double>(vecA[ind] * vecB[ind]);
  }
}

// #define gpuDotProduct_perk_pre(vecA,A_start,A_stop,A_step, vecB,B_start,B_stop,B_step, temp_sum)\
// do{\
//   _Pragma("unroll")\
//   for(int A_ind=A_start, B_ind=B_start; \
//     A_ind<A_stop&&B_ind<B_stop; \
//     A_ind+=A_step, B_ind+=B_step)\
//   {\
//     temp_sum += static_cast<double>(vecA[A_ind] * vecB[B_ind]);\
//   }\
// }while(0)


#define gpuDotProduct_perk_pre_reg(vecA, A_start, A_step, vecB,B_start, B_step, size, temp_sum)\
do{\
  _Pragma("unroll")\
   for(int i=0; i<size; i++)\
  {\
    int A_ind= A_start+i*A_step;\
    int B_ind= B_start+i*B_step;\
    if(A_ind>=N||B_ind>=N)break;\
    temp_sum += static_cast<double>(vecA[A_ind] * vecB[B_ind]);\
  }\
}while(0)

// template <typename        ValueT,
          // typename        OffsetT>
__device__ void gpuDotProduct_perk_post(double *result, double &temp_sum,
                                          const cg::thread_block &cta, 
                                          double * tmp, const cg::grid_group &grid) {
  cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

  temp_sum = cg::reduce(tile32, temp_sum, cg::plus<double>());

  if (tile32.thread_rank() == 0) {
    tmp[tile32.meta_group_rank()] = temp_sum;
  }
  // cg::sync(grid);
  cta.sync();

  if (tile32.meta_group_rank() == 0) {
    //32x32=1024(max of threaddim)
     temp_sum = tile32.thread_rank() < tile32.meta_group_size() ? tmp[tile32.thread_rank()] : 0.0;
     temp_sum = cg::reduce(tile32, temp_sum, cg::plus<double>());

    if (tile32.thread_rank() == 0) {
      atomicAdd(result, temp_sum);
    }
  }
}



__device__ void gpuCopyVector_perk(float *x, int x_start, int x_stop, int x_step,
                              float *y, int y_start, int y_stop, int y_step) {
  // for (int i = 0; i < size; i += step) 
  _Pragma("unroll")
  for(int x_ind=x_start, y_ind=y_start; 
    x_ind<x_stop&&y_ind<y_stop; 
    x_ind+=x_step, y_ind+=y_step)
  {
    y[y_ind] = x[x_ind];
  }
}





template<bool pre, bool post>
__device__ __forceinline__ void gpuSpMVDia_perk(float *val,// int num_rows,
                         float *inputVecX, 
                         float *outputVecY, int num_rows,//,const int start,const  int step,const  int stop, 
                         float alpha, cg::thread_block &cta, const cg::grid_group &grid
                        // ,float* sm_f
                        ) {
  // for (int i = start; i < stop; i += step) {
  for (int i = grid.thread_rank(); i < num_rows; i += grid.size()) {


    float output = 0.0;

    #pragma unroll
    for (int j = -1; j < 2; j++) {
      if(pre && i+j<0)continue;
      if(post && i+j>=num_rows)continue;
      
      // I or J or val arrays - can be put in shared memory
      output += alpha * val[3*i + j] * inputVecX[i+j];
    }

    outputVecY[i] = output;
  }
}

template<bool pre, bool post>
void __device__ gpuSpMVDiaSM_perk(float* sm_mat, int sm_start, int sm_step,
                                    float* inputVecX,
                                    float* outputVecY, int g_start ,int g_step, 
                                    int size,
                                    float alpha)
{
  for (int i = g_start, i_mat=sm_start; 
    i < size; 
    i += g_step, i_mat+=sm_step) {

    float output = 0.0;

    _Pragma("unroll")
    for (int j = -1; j < 2; j++) {
      if(true && i+j<0)continue;
      if(false && i+j>=size)continue;
      output += alpha * (sm_mat)[3*i_mat + j] * inputVecX[i+j];
    }
    outputVecY[i] = output;
  }
}




template <typename        ValueT,
          typename        OffsetT>
__device__ void gpuCopyVector_perk(ValueT *x, //int x_start, int x_stop, int x_step,
                              ValueT *y, OffsetT start, OffsetT stop, OffsetT step) {
  // for (int i = 0; i < size; i += step) 
  _Pragma("unroll")
  for(int ind=start; ind<stop; ind+=step)
  {
    y[ind] = x[ind];
  }
}

// #define gpuCopyVector_perk(x,x_start,x_stop,x_step,y,y_start,y_stop,y_step) \
do{\
  _Pragma("unroll")\
  for(int x_ind=x_start, y_ind=y_start; \
    x_ind<x_stop&&y_ind<y_stop; \
    x_ind+=x_step, y_ind+=y_step)\
  {\
    y[y_ind] = x[x_ind];\
  }\
}while(0) 


#define gpuCopyVector_perk_reg(x,x_start,x_step,y,y_start,y_step,size) \
do{\
  _Pragma("unroll")\
  for(int i=0; i<size; i++)\
  {\
    int y_ind= y_start+i*y_step;\
    int x_ind= x_start+i*x_step;\
    if(x_ind>=N||y_ind>=N)break;\
    y[y_ind] = x[x_ind];\
  }\
}while(0)



__device__ void gpuScaleVectorAndSaxpy_perk(float *x, int x_start, int x_stop, int x_step,
                              float *y, int y_start, int y_stop, int y_step,
                              float a, float scale) {
  // for (int i = 0; i < size; i += step) 
  _Pragma("unroll")
  for(int x_ind=x_start, y_ind=y_start; 
    x_ind<x_stop&&y_ind<y_stop; 
    x_ind+=x_step, y_ind+=y_step)
  {
    y[y_ind] = a * x[x_ind] + scale * y[y_ind];
  }
}
template <typename        ValueT,
          typename        OffsetT>
__device__ void gpuScaleVectorAndSaxpy_perk(ValueT *x, 
                              ValueT *y, OffsetT start, OffsetT stop, OffsetT step,
                              ValueT a, ValueT scale) {
  // for (int i = 0; i < size; i += step) 
  _Pragma("unroll")
  for(OffsetT ind=start; ind<stop; ind+=step)
  {
    y[ind] = a * x[ind] + scale * y[ind];
  }
}


// #define gpuScaleVectorAndSaxpy_perk(x,x_start,x_stop,x_step,y,y_start,y_stop,y_step  , a, scale) \
// do{\
//   _Pragma("unroll")\
//   for(int x_ind=x_start, y_ind=y_start; \
//     x_ind<x_stop&&y_ind<y_stop; \
//     x_ind+=x_step, y_ind+=y_step)\
//   {\
//     y[y_ind] = a * x[x_ind] + scale * y[y_ind];\
//   }\
// }while(0) 



#define gpuScaleVectorAndSaxpy_perk_reg(x,x_start,x_step,   \
                                        y,y_start,y_step,   \
                                        size, \
                                        a, scale) \
do{\
  _Pragma("unroll")\
  for(int i=0; i<size; i++)\
  {\
    int y_ind= y_start+i*y_step;\
    int x_ind= x_start+i*x_step;\
    if(x_ind>=N||y_ind>=N)break;\
    y[y_ind] = a * x[x_ind] + scale * y[y_ind];\
  }\
}while(0) 



