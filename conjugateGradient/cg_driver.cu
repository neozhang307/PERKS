

/*
 * Copyright 1993-2017 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 * This sample implements a conjugate gradient solver on GPU using
 * Multi Block Cooperative Groups, also uses Unified Memory.
 *
//  */
#define THRUST_IGNORE_CUB_VERSION_CHECK
// #include <map>



#include "cg.h"
#include "util/timer.cuh"
#include "util/cub_utils.cuh"

// #include "util/command.cuh"
// #include <cub_utils.cuh>

#include "cg_dispatcher.cuh"

#include <helper_cuda.h>  // helper function CUDA error checking and initialization
#include <helper_functions.h>  // helper for shared functions common to CUDA Samples
#include <stdio.h>


using namespace cub;





bool                    g_quiet     = false;        // Whether to display stats in CSV format
// bool                    g_verbose   = false;        // Whether to display output to console
// bool                    g_verbose2  = false;        // Whether to display input to console
CachingDeviceAllocator  g_allocator(true);          // Caching allocator for device memory

// #undef USEVDATA
#include "tridiag.h"
#include "cg_driver.cuh"

template <
    typename ValueT,
    typename OffsetT>
void myTest(
  int devID,
  cudaDeviceProp& deviceProp,
  CommandLineArgs& args){
  int N = 0, nz = 0, *I = NULL, *J = NULL;
  ValueT *val = NULL;
  // const ValueT tol = 1e-10f;
  const ValueT tol = 1e-11f;
  ValueT *x;
  ValueT *rhs;
  ValueT r1;
  ValueT *r, *p, *Ax;
  cudaEvent_t start, stop;

  printf("Starting ...\n");

  // This will pick the best possible CUDA capable device

  if (!deviceProp.managedMemory) {
    // This sample requires being run on a device that supports Unified Memory
    fprintf(stderr, "Unified Memory not supported on this device\n");
    exit(EXIT_WAIVED);
  }

  // This sample requires being run on a device that supports Cooperative Kernel
  // Launch
  if (!deviceProp.cooperativeLaunch) {
    printf(
        "\nSelected GPU (%d) does not support Cooperative Kernel Launch, "
        "Waiving the run\n",
        devID);
    exit(EXIT_WAIVED);
  }

  // Statistics about the GPU device
  printf(
      "> GPU device has %d Multi-Processors, SM %d.%d compute capabilities\n\n",
      deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);

  /* Generate a random tridiagonal symmetric matrix in CSR format */
#ifdef USEVDATA
  N = 1048576;//*8;
  nz = (N - 2) * 3 + 4;
  int                 max_iter   = N;
  std::string         mtx_filename;
#else

  std::string         mtx_filename;
  args.GetCmdLineArgument("mtx", mtx_filename);

  if(mtx_filename=="")
  {
      mtx_filename="/home/Lingqi/workspace/merge_spmv/perk_cg/data/bmwcra_1/bmwcra_1.mtx";
      // mtx_filename="/home/Lingqi/data/general/fv1/fv1.mtx";
  }
  // printf("opening %s\n",mtx_filename.get_ptr());
  // cout<<"opening "<<mtx_filename;
  int                 max_iter   = -1;
  args.GetCmdLineArgument("iters", max_iter);



  CooMatrix<ValueT, OffsetT> coo_matrix;
  coo_matrix.InitMarket(mtx_filename, 1.0, !g_quiet);

  max_iter= (max_iter==-1?coo_matrix.num_rows*10:max_iter);
  printf("maxiter is %d\n",max_iter);
  // coo_matrix.InitMarket("/home/Lingqi/workspace/merge_spmv/perk_cg/data/bmwcra_1/bmwcra_1.mtx", 1.0, !g_quiet);
  printf("<%d,%d>\n",coo_matrix.num_rows,coo_matrix.num_nonzeros);
  CsrMatrix<ValueT, OffsetT> csr_matrix(coo_matrix);
  N=coo_matrix.num_rows;
  nz=coo_matrix.num_nonzeros;
#endif
  cudaMallocManaged(reinterpret_cast<void **>(&I), sizeof(OffsetT) * (N + 1));
  cudaMallocManaged(reinterpret_cast<void **>(&J), sizeof(OffsetT) * nz);
  cudaMallocManaged(reinterpret_cast<void **>(&val), sizeof(ValueT) * nz);

#ifndef USEVDATA
  cudaMemcpy(val, csr_matrix.values, nz*sizeof(ValueT), cudaMemcpyHostToDevice);
  cudaMemcpy(I, csr_matrix.row_offsets, (N+1)*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(J, csr_matrix.column_indices, nz*sizeof(int), cudaMemcpyHostToDevice);
#else
///////
  genTridiag(I, J, val, N, nz);
#endif

  cudaMallocManaged(reinterpret_cast<void **>(&x), sizeof(ValueT) * N);
 
  cudaMallocManaged(reinterpret_cast<void **>(&rhs), sizeof(ValueT) * N);

  double *dot_result;
  cudaMallocManaged(reinterpret_cast<void **>(&dot_result), sizeof(double)*2);

  *dot_result = 0.0;

  // temp memory for CG
  checkCudaErrors(
      cudaMallocManaged(reinterpret_cast<void **>(&r), N * sizeof(ValueT)));
  checkCudaErrors(
      cudaMallocManaged(reinterpret_cast<void **>(&p), N * sizeof(ValueT)));
  checkCudaErrors(
      cudaMallocManaged(reinterpret_cast<void **>(&Ax), (N+1) * sizeof(ValueT)));
  cudaDeviceSynchronize();
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));


#if ENABLE_CPU_DEBUG_CODE
  ValueT *Ax_cpu = reinterpret_cast<ValueT *>(malloc(sizeof(ValueT) * N));
  ValueT *r_cpu = reinterpret_cast<ValueT *>(malloc(sizeof(ValueT) * N));
  ValueT *p_cpu = reinterpret_cast<ValueT *>(malloc(sizeof(ValueT) * N));
  ValueT *x_cpu = reinterpret_cast<ValueT *>(malloc(sizeof(ValueT) * N));

  for (int i = 0; i < N; i++) {
    r_cpu[i] = 1.0;
    Ax_cpu[i] = x_cpu[i] = 0.0;
  }

#endif

  for (int i = 0; i < N; i++) {
    r[i] = rhs[i] = 1.0;
    // x[i] = 1;
    x[i] = 0.0;
  }



  CgParams<ValueT, double, OffsetT> cgParamsT;
  cgParamsT.d_val            = val;
  cgParamsT.d_row_offsets    = I;
  cgParamsT.d_column_indices = J;
  cgParamsT.d_vector_x       = x;
  cgParamsT.d_vector_Ax      = Ax;
  cgParamsT.d_vector_p       = p;
  cgParamsT.d_vector_r       = r;
  cgParamsT.d_vector_dot_results=dot_result;


  cgParamsT.N                = N;
  cgParamsT.nz               = nz;
  cgParamsT.tol              = tol;
  cgParamsT.max_iter              = 2;
  cudaMalloc((void**)&cgParamsT.d_iter, sizeof(unsigned int));

    ValueT* d_y;
    cudaMalloc((void**)&d_y, sizeof(ValueT)*(N+1));
  SpmvParams<ValueT, OffsetT> params;
  // CubDebugExit(g_allocator.DeviceAllocate((void **) &params.d_values,          sizeof(ValueT) * csr_matrix.num_nonzeros));
  // CubDebugExit(g_allocator.DeviceAllocate((void **) &params.d_row_end_offsets, sizeof(OffsetT) * (csr_matrix.num_rows + 1)));
  // CubDebugExit(g_allocator.DeviceAllocate((void **) &params.d_column_indices,  sizeof(OffsetT) * csr_matrix.num_nonzeros));
  // CubDebugExit(g_allocator.DeviceAllocate((void **) &params.d_vector_x,        sizeof(ValueT) * csr_matrix.num_cols));
  // CubDebugExit(g_allocator.DeviceAllocate((void **) &params.d_vector_y,        sizeof(ValueT) * csr_matrix.num_rows));
  params.d_values         = val;
  params.d_row_end_offsets= I+1;
  params.d_column_indices = J;
  params.d_vector_x       = p;
  params.d_vector_y       = Ax;

  params.num_rows         = N;
  params.num_cols         = N;
  params.num_nonzeros     = nz;
  params.alpha            = 1.0;//alpha;
  params.beta             = 0;

  size_t temp_storage_bytes = 0;
  void *d_temp_storage = NULL;


    // REAL l2perused;
    size_t inner_window_size = N*sizeof(ValueT);//30*1024*1024;
    cudaStreamAttrValue stream_attribute;
    stream_attribute.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(p);                  // Global Memory data pointer
    stream_attribute.accessPolicyWindow.num_bytes = inner_window_size;                                   // Number of bytes for persistence access
    stream_attribute.accessPolicyWindow.hitRatio  = 1;                                             // Hint for cache hit ratio
    stream_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;                  // Persistence Property
    stream_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;  

    cudaStreamSetAttribute(0, cudaStreamAttributeAccessPolicyWindow, &stream_attribute); 
    cudaCtxResetPersistingL2Cache();
    cudaStreamSynchronize(0);

  // mycsrMatrix.num_rows=N;
  // mycsrMatrix.num_cols=N;
  // mycsrMatrix.num_nonzeros=nz;
  // mycsrMatrix.row_offsets=I;
  // mycsrMatrix.column_indices=J;
  // mycsrMatrix.values=val;

  // // Get amount of temporary storage needed
    CubDebugExit(
        (DispatchCG<ValueT,OffsetT>::InitDispatch(
            d_temp_storage,
            temp_storage_bytes,
            params,
            0, false))
        );
    printf("temp size is %lu\n",temp_storage_bytes);

    // // Warmup
  // Allocate
    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));



    // Reset input/output vector y
    CubDebugExit(cudaMemcpy(params.d_vector_y, p, sizeof(ValueT) * params.num_rows, cudaMemcpyHostToDevice));
    SMCacheParams<OffsetT,size_t> smParamsT;
    // Warmup
     DispatchCG<ValueT,OffsetT>::ProcessDispatchCG(smParamsT,cgParamsT, 
                                        d_temp_storage, temp_storage_bytes,params);
  
     cgParamsT.max_iter              =  max_iter;//*10;//10000;//N;//10000;//57356;//31994;//N;//57356;//31994;//N;//10000;//N;//N;//10000;//N;//10000;//N;//1000;//N;//10000;//57356;//N;

  checkCudaErrors(cudaEventRecord(start, 0));


  DispatchCG<ValueT,OffsetT>::ProcessDispatchCG(smParamsT,cgParamsT, 
                                        d_temp_storage, temp_storage_bytes,params);
  
  checkCudaErrors(cudaEventRecord(stop, 0));
  checkCudaErrors(cudaDeviceSynchronize());

  float time;
  checkCudaErrors(cudaEventElapsedTime(&time, start, stop));

  r1 = *dot_result;
  unsigned int total_iter;
  cudaMemcpy(&total_iter, cgParamsT.d_iter, sizeof(unsigned int), cudaMemcpyDeviceToHost);
  printf("GPU Final, residual = %e, kernel execution time = %f ms in %d iteration\n", sqrt(r1),
         time,total_iter);

  fprintf(stderr,"%s\t"
    "%e\t"
    "%d\t%d\t%d\t"
    "%lu\t"
    // "%d\t%d\t%d\t%d\t"
    "%d\t%d\t"
    "%f\t%d\n",
        mtx_filename.c_str(),
        sqrt(r1),
        nz, N, cgParamsT.gridDim,
        smParamsT.sMemSizeTemptTotal,
        // smParamsT.sm_size_total_coor, 
        // smParamsT.sm_size_total_thread_coor, 
        // smParamsT.sm_size_total_vals, 
        // smParamsT.sm_size_total_cols, 
        smParamsT.sm_num_r, 
        smParamsT.sm_blk_size_r, 

        time, total_iter);
  size_t totalaccess=nz*(sizeof(ValueT)*2+sizeof(OffsetT))+N*12*sizeof(ValueT)+N*sizeof(OffsetT);
  printf("<%f ms: %fGB/s>\n",time/total_iter,(double)totalaccess/time*total_iter*1000/1024/1024/1024);
  

#if ENABLE_CPU_DEBUG_CODE
  cpuConjugateGrad(I, J, val, x_cpu, Ax_cpu, p_cpu, r_cpu, nz, N, tol);
#endif

  // for(int i=0; i<5; i++)
  // {
  //   printf("<%f>\n",x[i]);
  // }

  ValueT rsum, diff, err = 0.0;

  for (int i = 0; i < N; i++) {
    rsum = 0.0;

    for (int j = I[i]; j < I[i + 1]; j++) {
      rsum += val[j] * x[J[j]];
    }

    diff = fabs(rsum - rhs[i]);

    if (diff > err) {
      err = diff;
    }
  }

  checkCudaErrors(cudaFree(I));
  checkCudaErrors(cudaFree(J));
  checkCudaErrors(cudaFree(val));
  checkCudaErrors(cudaFree(x));
  checkCudaErrors(cudaFree(rhs));
  checkCudaErrors(cudaFree(r));
  checkCudaErrors(cudaFree(p));
  checkCudaErrors(cudaFree(Ax));
  checkCudaErrors(cudaFree(dot_result));
  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));

#if ENABLE_CPU_DEBUG_CODE
  free(Ax_cpu);
  free(r_cpu);
  free(p_cpu);
  free(x_cpu);
#endif

  printf("Test Summary:  Error amount = %f \n", err);
  fprintf(stdout, "&&&& conjugateGradientMultiBlockCG %s\n",
          (sqrt(r1) < tol) ? "PASSED" : "FAILED");
}


template void myTest<float,int>(int devID,cudaDeviceProp& deviceProp,CommandLineArgs& args);