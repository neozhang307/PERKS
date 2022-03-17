
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

  bool isbaseline=true;
  // bool usecoo=false;
  bool cachematrix=false;
  bool cachevector=false;

  isbaseline = args.CheckCmdLineFlag("baseline");
  cachematrix = args.CheckCmdLineFlag("cmat");
  cachevector = args.CheckCmdLineFlag("cvec");
  // if(!isbaseline&&!cachevector&&!cachematrix)
  if(isbaseline)
  {
    // usecoo=false;
    cachematrix=false;
    cachevector=false;
  }
  // printf("%d",fp32);
  if (fp32)
  {
    if(isbaseline)
    {
      myTest<float,int,true,false,false>(devID,deviceProp,args);
    }
    else
    {
      if(cachematrix&&cachevector)
      {
        myTest<float,int,false,true,true>(devID,deviceProp,args);
      }
      else if(cachematrix&&!cachevector)
      {
        myTest<float,int,false,true,false>(devID,deviceProp,args);
      }
      else if(!cachematrix&&cachevector)
      {
        myTest<float,int,false,false,true>(devID,deviceProp,args);
      }
      else if(!cachematrix&&!cachevector)
      {
        myTest<float,int,false,false,false>(devID,deviceProp,args);
      }
    }
  }
  else
  {
    if(isbaseline)
    {
      myTest<double,int,true,false,false>(devID,deviceProp,args);
    }
    else
    {
      if(cachematrix&&cachevector)
      {
        myTest<double,int,false,true,true>(devID,deviceProp,args);
      }
      else if(cachematrix&&!cachevector)
      {
        myTest<double,int,false,true,false>(devID,deviceProp,args);
      }
      else if(!cachematrix&&cachevector)
      {
        myTest<double,int,false,false,true>(devID,deviceProp,args);
      }
      else if(!cachematrix&&!cachevector)
      {
        myTest<double,int,false,false,false>(devID,deviceProp,args);
      }
    }
  }
  // exit((sqrt(r1) < tol) ? EXIT_SUCCESS : EXIT_FAILURE);
}


