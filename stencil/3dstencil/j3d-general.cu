#ifndef CONFIGURE
  #include "./config.cuh"
  #include "./genconfig.cuh"
#endif
#include "./common/cuda_computation.cuh"
#include "./common/cuda_common.cuh"
#include "./common/types.hpp"
#include <math.h>

#include <cooperative_groups.h>

#ifdef ASYNCSM
  // #if PERKS_ARCH<800 
    // #error "unsupport architecture"
  // #endif
  #include <cooperative_groups/memcpy_async.h>
  #include <cuda_pipeline.h>
#endif

#ifdef USESM
#define isUseSM (true)
#else
#define isUseSM (false)
#endif
namespace cg = cooperative_groups;

// #define NOCACHE_Y (0)
#define NOCACHE_Z (HALO)
// #define LOCAL_TILE_Y (TILE_Y-2*NOCACHE_Y)
#include "./j3d-general-kernels.cuh"


#define MAXTHREAD (256)
#define MINBLOCK (1)

template<class REAL, int halo, 
int LOCAL_ITEM_PER_THREAD, int LOCAL_TILE_X, int LOCAL_TILE_Y, const int reg_folder_z,int minblocks, bool UseSMCache >
// __launch_bounds__(256, 2)
__launch_bounds__(MAXTHREAD, minblocks)
__global__ void 
kernel3d_general(REAL * __restrict__ input, 
                                REAL * __restrict__ output, 
                                int width_z, int width_y, int width_x,
                                REAL* l2_cache_i, REAL* l2_cache_o,
                                int iteration,
                                int max_sm_flder) 
{
  kernel3d_general_inner<REAL,halo,LOCAL_ITEM_PER_THREAD,LOCAL_TILE_X,LOCAL_TILE_Y,reg_folder_z,UseSMCache>
  (input,output,width_z,width_y,width_x,l2_cache_i,l2_cache_o,iteration,max_sm_flder);
}


// template __global__ void kernel3d_general<float,HALO,ITEM_PER_THREAD,TILE_X,TILE_Y,0,false> 
//     (float *__restrict__, float *__restrict__ , int , int , int, float*,float*,int,int);

// template __global__ void kernel3d_general<float,HALO,ITEM_PER_THREAD,TILE_X,TILE_Y,0,true> 
//     (float *__restrict__, float *__restrict__ , int , int , int, float*,float*,int,int);

// template __global__ void kernel3d_general<float,HALO,ITEM_PER_THREAD,TILE_X,TILE_Y,REG_FOLDER_Z,false> 
//     (float *__restrict__, float *__restrict__ , int , int , int, float*,float*,int,int);

// template __global__ void kernel3d_general<float,HALO,ITEM_PER_THREAD,TILE_X,TILE_Y,REG_FOLDER_Z,true> 
//     (float *__restrict__, float *__restrict__ , int , int , int, float*,float*,int,int);

// template __global__ void kernel3d_general<double,HALO,ITEM_PER_THREAD,TILE_X,TILE_Y,REG_FOLDER_Z,true> 
//     (double *__restrict__, double *__restrict__ , int , int , int , double*,double*,int,int);
#ifndef CONFIGURE
  PERKS_INITIALIZE_ALL_TYPE_7ARG(PERKS_DECLARE_INITIONIZATION_GENERAL,HALO,ITEM_PER_THREAD,TILE_X,TILE_Y,REG_FOLDER_Z,MINBLOCK,true);
  PERKS_INITIALIZE_ALL_TYPE_7ARG(PERKS_DECLARE_INITIONIZATION_GENERAL,HALO,ITEM_PER_THREAD,TILE_X,TILE_Y,REG_FOLDER_Z,MINBLOCK,false);
#else
  template __global__ void kernel3d_general<TYPE,HALO,ITEM_PER_THREAD,TILE_X,TILE_Y,REG_FOLDER_Z,BLOCKTYPE,isUseSM> 
    (TYPE *__restrict__, TYPE *__restrict__ , int , int , int, TYPE*,TYPE*,int,int);
#endif
// #ifndef PERSISTENT 
  // PERKS_INITIALIZE_ALL_TYPE_1ARG(PERKS_DECLARE_INITIONIZATION_BASELINE,HALO);
// #else
  // PERKS_INITIALIZE_ALL_TYPE_1ARG(PERKS_DECLARE_INITIONIZATION_PERSISTENT,HALO);
// #endif