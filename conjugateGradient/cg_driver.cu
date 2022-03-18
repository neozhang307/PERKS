

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
    typename OffsetT,
    bool baseline,
    bool cacheMatrix,
    bool cacheVector>
void myTest(
  int devID,
  cudaDeviceProp& deviceProp,
  CommandLineArgs& args){
  int N = 0, nz = 0, *I = NULL, *J = NULL;
  ValueT *val = NULL;
  const ValueT tol = 1e-10f;
  // const ValueT tol = 1e-200f;
  // const ValueT tol = 0;
  ValueT *x;
  ValueT *rhs;
  ValueT r1;
  ValueT *r, *p, *Ax;
  cudaEvent_t start, stop;
  // printf("----");
#ifndef __PRINT__
  printf("Starting ...\n");
#endif
  // This will pick the best possible CUDA capable device

  if (!deviceProp.managedMemory) {
    // This sample requires being run on a device that supports Unified Memory
    #ifndef __PRINT__
    fprintf(stderr, "Unified Memory not supported on this device\n");
    #endif
    exit(EXIT_WAIVED);
  }
  // Launch
  if (!deviceProp.cooperativeLaunch) {
    #ifndef __PRINT__
    printf(
        "\nSelected GPU (%d) does not support Cooperative Kernel Launch, "
        "Waiving the run\n",
        devID);
    #endif
    exit(EXIT_WAIVED);
  }

  // Statistics about the GPU device
  #ifndef __PRINT__
  printf(
      "> GPU device has %d Multi-Processors, SM %d.%d compute capabilities\n\n",
      deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);
  #endif

  std::string         mtx_filename;
  bool usevdata=false;
  
  // bool baseline=true;
  // bool usecoo=false;
  // bool cachematrix=false;
  // bool cachevector=false;

  bool warmup=false;
  int  max_iter   = -1;
  bool isCheck=false;
  usevdata = args.CheckCmdLineFlag("vdata");
  warmup = args.CheckCmdLineFlag("warmup");
  isCheck = args.CheckCmdLineFlag("check");
  // baseline = args.CheckCmdLineFlag("baseline");
  // cachematrix = args.CheckCmdLineFlag("cmat");
  // cachevector = args.CheckCmdLineFlag("cvec");
  // if(baseline)
  // {
  //   usecoo=false;
  //   cachematrix=false;
  //   cachevector=false;
  // }

  args.GetCmdLineArgument("iters", max_iter);
  // printf("usedata%d\n",usevdata);
// #ifdef USEVDATA
  if(usevdata)
  {
    N = 1048576;//*8;
    nz = (N - 2) * 3 + 4;
    max_iter   = N;
    mtx_filename="virtualdata";

    cudaMallocManaged(reinterpret_cast<void **>(&I), sizeof(OffsetT) * (N + 1));
    cudaMallocManaged(reinterpret_cast<void **>(&J), sizeof(OffsetT) * nz);
    cudaMallocManaged(reinterpret_cast<void **>(&val), sizeof(ValueT) * nz);

    genTridiag(I, J, val, N, nz);
  }
  else
  {
    args.GetCmdLineArgument("mtx", mtx_filename);

    if(mtx_filename=="")
    {
        mtx_filename="/home/Lingqi/data/general/bmwcra_1/bmwcra_1.mtx";
        
    }
    
    

    CooMatrix<ValueT, OffsetT> coo_matrix;
    coo_matrix.InitMarket(mtx_filename, 1.0, !g_quiet);
    max_iter= (max_iter<=0?coo_matrix.num_rows*10:max_iter);

    CsrMatrix<ValueT, OffsetT> csr_matrix(coo_matrix);
    N=coo_matrix.num_rows;
    nz=coo_matrix.num_nonzeros;

    cudaMallocManaged(reinterpret_cast<void **>(&I), sizeof(OffsetT) * (N + 1));
    cudaMallocManaged(reinterpret_cast<void **>(&J), sizeof(OffsetT) * nz);
    cudaMallocManaged(reinterpret_cast<void **>(&val), sizeof(ValueT) * nz);

    cudaMemcpy(val, csr_matrix.values, nz*sizeof(ValueT), cudaMemcpyHostToDevice);
    cudaMemcpy(I, csr_matrix.row_offsets, (N+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(J, csr_matrix.column_indices, nz*sizeof(int), cudaMemcpyHostToDevice);   
  }

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


  // // Get amount of temporary storage needed
    CubDebugExit(
        (DispatchCG<ValueT,OffsetT,baseline,cacheMatrix,cacheVector>::InitDispatch(
            d_temp_storage,
            temp_storage_bytes,
            params,
            0, false))
        );
#ifndef __PRINT__
    printf("temp size is %lu\n",temp_storage_bytes);
#endif
    // // Warmup
  // Allocate
  CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));



    // Reset input/output vector y
  CubDebugExit(cudaMemcpy(params.d_vector_y, p, sizeof(ValueT) * params.num_rows, cudaMemcpyHostToDevice));
  SMCacheParams<OffsetT,size_t> smParamsT;
  // Warmup
  if(warmup)
  {
    printf("warmup start\n");
    cudaEvent_t warmup_start, warmup_stop;
    checkCudaErrors(cudaEventCreate(&warmup_start));
    checkCudaErrors(cudaEventCreate(&warmup_stop));
    checkCudaErrors(cudaEventRecord(warmup_start, 0));
    DispatchCG<ValueT,OffsetT,baseline,cacheMatrix,cacheVector>::ProcessDispatchCG(smParamsT,cgParamsT, 
                                        d_temp_storage, temp_storage_bytes,params);    
    checkCudaErrors(cudaEventRecord(warmup_stop, 0));
    checkCudaErrors(cudaDeviceSynchronize());
    float time;
    checkCudaErrors(cudaEventElapsedTime(&time, warmup_start, warmup_stop));
    cgParamsT.max_iter              =max(1, (int) (2*350/time));
    DispatchCG<ValueT,OffsetT,baseline,cacheMatrix,cacheVector>::ProcessDispatchCG(smParamsT,cgParamsT, 
                                        d_temp_storage, temp_storage_bytes,params);        
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaEventDestroy(warmup_start));
    checkCudaErrors(cudaEventDestroy(warmup_stop));
    printf("finished warmup \n");
  }

  
  cgParamsT.max_iter              =  max_iter;//*10;//10000;//N;//10000;//57356;//31994;//N;//57356;//31994;//N;//10000;//N;//N;//10000;//N;//10000;//N;//1000;//N;//10000;//57356;//N;
  printf("max iter is %u\n",cgParamsT.max_iter );
  checkCudaErrors(cudaEventRecord(start, 0));
  // printf("----");

  DispatchCG<ValueT,OffsetT,baseline,cacheMatrix,cacheVector>::ProcessDispatchCG(smParamsT,cgParamsT, 
                                        d_temp_storage, temp_storage_bytes,params);
  
  checkCudaErrors(cudaEventRecord(stop, 0));
  checkCudaErrors(cudaDeviceSynchronize());

  float time;
  checkCudaErrors(cudaEventElapsedTime(&time, start, stop));
  // printf("----");
  r1 = *dot_result;
  unsigned int total_iter;
  cudaMemcpy(&total_iter, cgParamsT.d_iter, sizeof(unsigned int), cudaMemcpyDeviceToHost);
#ifndef __PRINT__
  printf("GPU Final, residual = %e, kernel execution time = %f ms in %d iteration\n", sqrt(r1),
         time,total_iter);
#endif
  printf("max iter is %d\n",max_iter);
// #ifdef __PRINT__
  // mtx_filename="nothing";
  // printf("%shaha\n",mtx_filename.c_str());
  printf("%s\t",mtx_filename.c_str());
  fprintf(stderr,"%s\t",mtx_filename.c_str());
  fprintf(stderr,"%d\t",sizeof(ValueT)/4);
  if(baseline)
  {
    fprintf(stderr,"bsln\t");
  }
  else if(cacheMatrix&&cacheVector)
  {
    fprintf(stderr,"mix\t");
  }
  else if(!cacheMatrix&&!cacheVector)
  {
    fprintf(stderr,"coo\t");
  }
  else if(!cacheMatrix&&cacheVector)
  {
    fprintf(stderr,"cvec\t");
  }
  else if(cacheMatrix&&!cacheVector)
  {
    fprintf(stderr,"cmat\t");
  }
  fprintf(stderr,"%e\t",sqrt(r1));
  
  fprintf(stderr,"%d\t%d\t",nz, N);
  fprintf(stderr,"%d\t%d\t",cgParamsT.gridDim, THREADS_PER_BLOCK); //blockdim
  fprintf(stderr,"%d\t%d\t",smParamsT.sm_num_r, smParamsT.sm_num_matrixperblk);
  fprintf(stderr,"%f\t",(double)(nz*(sizeof(ValueT)+sizeof(OffsetT))+N*sizeof(ValueT)+N*sizeof(OffsetT))/1024/1024);
  fprintf(stderr,"%f\t%f\t%f\t",(double)smParamsT.sm_size_coor/1024, (double)smParamsT.sm_blk_size_r/1024, (double)smParamsT.sm_size_unit_matrix*smParamsT.sm_num_matrixperblk/1024);
  fprintf(stderr,"%f\t",(double)smParamsT.sMemSize/1024);
  fprintf(stderr,"%d\t%f\t",total_iter,time);
  // size_t spmvaccess= nz*sizeof(ValueT)*2+(nz+N+1)*sizeof(OffsetT);
  // size_t totalaccess=spmvaccess+7*N*sizeof(ValueT)+max_iter*(spmvaccess+9*N*sizeof(ValueT));
  //MINIAN mem access: 1 spmv load + 3 vector update(1 update = 1 load + 1 store)
  size_t spmvaccess= nz*sizeof(ValueT)+(nz+N+1)*sizeof(OffsetT);
  size_t totalaccess=max_iter*(spmvaccess+6*N*sizeof(ValueT));
  // 2*N+3*N + 3*N*(max_iter-1)+ max_iter*(2*N+3*N+2*N    2*nnz+(1+nnz+N))
  // size_t totalaccess=nz*(sizeof(ValueT)*2+sizeof(OffsetT))+N*9*sizeof(ValueT)+N*sizeof(OffsetT);
  fprintf(stderr,"%f\t%f\t\n",time/total_iter,(double)totalaccess/time*1000/1024/1024/1024);
// #endif
// #ifndef __PRINT__
  // size_t totalaccess=nz*(sizeof(ValueT)*2+sizeof(OffsetT))+N*12*sizeof(ValueT)+N*sizeof(OffsetT);
  printf("<%f ms: %fGB/s>\n",time/total_iter,(double)totalaccess/time*1000/1024/1024/1024);
// #endif
//need to print
#if ENABLE_CPU_DEBUG_CODE
  cpuConjugateGrad(I, J, val, x_cpu, Ax_cpu, p_cpu, r_cpu, nz, N, tol);
#endif

  // for(int i=0; i<5; i++)
  // {
  //   printf("<%f>\n",x[i]);
  // }

  ValueT rsum, diff, err = 0.0;
  if(isCheck)
  {

    for (int i = 0; i < N; i++) {
      rsum = 0.0;

      for (int j = I[i]; j < I[i + 1]; j++) {
        rsum += val[j] * x[J[j]];
      }

      diff = fabs(rsum - rhs[i]);

      if (diff > err) {
        err = diff;
        // printf("<%d:%f>",i,rsum);
      }
    }
    printf("* Test Summary:  Error amount = %f \n", err);

  }
  // ValueT * cpu_ptr=(ValueT*)malloc(sizeof(ValueT)*N);
  // cudaMemcpy(cpu_ptr,x,N*sizeof(ValueT),cudaMemcpyDeviceToHost);
  // for(in)
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

#ifndef __PRINT__
  printf("Test Summary:  Error amount = %f \n", err);
  fprintf(stdout, "&&&& conjugateGradientMultiBlockCG %s\n",
          (sqrt(r1) < tol) ? "PASSED" : "FAILED");
#endif
}

#ifndef COMPILE
template void myTest<float,int,true,false,false>(int devID,cudaDeviceProp& deviceProp,CommandLineArgs& args);
template void myTest<double,int,true,false,false>(int devID,cudaDeviceProp& deviceProp,CommandLineArgs& args);

template void myTest<float,int,false,false,true>(int devID,cudaDeviceProp& deviceProp,CommandLineArgs& args);
template void myTest<double,int,false,false,true>(int devID,cudaDeviceProp& deviceProp,CommandLineArgs& args);

template void myTest<float,int,false,true,false>(int devID,cudaDeviceProp& deviceProp,CommandLineArgs& args);
template void myTest<double,int,false,true,false>(int devID,cudaDeviceProp& deviceProp,CommandLineArgs& args);

template void myTest<float,int,false,true,true>(int devID,cudaDeviceProp& deviceProp,CommandLineArgs& args);
template void myTest<double,int,false,true,true>(int devID,cudaDeviceProp& deviceProp,CommandLineArgs& args);

template void myTest<float,int,false,false,false>(int devID,cudaDeviceProp& deviceProp,CommandLineArgs& args);
template void myTest<double,int,false,false,false>(int devID,cudaDeviceProp& deviceProp,CommandLineArgs& args);
#else 
template void myTest<REAL,int,ISBASE,CMAT,CVEC>(int devID,cudaDeviceProp& deviceProp,CommandLineArgs& args);

#endif