
#include <helper_cuda.h>  // helper function CUDA error checking and initialization
#include <helper_functions.h>  // helper for shared functions common to CUDA Samples

#include "util/cub_utils.cuh"
// #include <cub_utils.cuh>
#include "cg_driver.cuh"

// #include "util/command.cuh"

int main(int argc, char **argv) {
  CommandLineArgs args(argc, argv);

  bool                fp32=true;
  int devID = findCudaDevice(argc, (const char **)argv);
  cudaDeviceProp deviceProp;
  checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));
  fp32 = args.CheckCmdLineFlag("fp32");
  printf("%d",fp32);
  // if (fp32)
  {
    printf("float\n");
    fprintf(stderr,"float\t");
    myTest<float,int>(devID,deviceProp,args);
  }
  // else
  // {
  //   printf("double\n");
  //   fprintf(stderr,"double\t");
  //   myTest<double,int>(devID,deviceProp,args);
  // }
  // exit((sqrt(r1) < tol) ? EXIT_SUCCESS : EXIT_FAILURE);
}


