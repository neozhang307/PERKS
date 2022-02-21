#ifndef CONFIGURE
  #include "./config.cuh"
  #include "./genconfig.cuh"
#endif
// #include "./common/cuda_common.cuh"
// #include "./common/cuda_computation.cuh"
// #include "./common/types.hpp"
// #include <math.h>

#include <cooperative_groups.h>

#ifdef SMASYNC
  #include <cooperative_groups/memcpy_async.h>
  #include <cuda_pipeline.h>
#endif

namespace cg = cooperative_groups;

#include "./jacobi-general-kernel.cuh"

#define MAXTHREAD (256)
#define MINBLOCK (1)
template<class REAL, int LOCAL_TILE_Y, int halo, int reg_folder_y, int minblocks, bool UseSMCache>
__launch_bounds__(MAXTHREAD, minblocks)
__global__ void 
#ifndef BOX
  #ifdef SMASYNC
    kernel_general_async
  #else 
    kernel_general
  #endif
#else
  #ifdef SMASYNC
    kernel_general_box_async
  #else 
    kernel_general_box
  #endif
#endif
(REAL * __restrict__ input, int width_y, int width_x, 
  REAL * __restrict__ __var_4__, 
  REAL * __restrict__ l2_cache_o,REAL * __restrict__ l2_cache_i,
  int iteration,
  int max_sm_flder)
{
  inner_general<REAL, LOCAL_TILE_Y, halo, reg_folder_y, UseSMCache>( input,  width_y,  width_x, 
    __var_4__, 
    l2_cache_o, l2_cache_i,
    iteration,
    max_sm_flder);
}

#ifndef CONFIGURE
#ifndef SMASYNC
  #ifndef BOX
      PERKS_INITIALIZE_ALL_TYPE_5ARG(PERKS_DECLARE_INITIONIZATION_GENERAL,8,HALO,REG_FOLDER_Y,MINBLOCK,true);
      PERKS_INITIALIZE_ALL_TYPE_5ARG(PERKS_DECLARE_INITIONIZATION_GENERAL,8,HALO,REG_FOLDER_Y,MINBLOCK,false);
      PERKS_INITIALIZE_ALL_TYPE_5ARG(PERKS_DECLARE_INITIONIZATION_GENERAL,16,HALO,REG_FOLDER_Y,MINBLOCK,true);
      PERKS_INITIALIZE_ALL_TYPE_5ARG(PERKS_DECLARE_INITIONIZATION_GENERAL,16,HALO,REG_FOLDER_Y,MINBLOCK,false);
  #else
      PERKS_INITIALIZE_ALL_TYPE_5ARG(PERKS_DECLARE_INITIONIZATION_GENERALBOX,8,HALO,REG_FOLDER_Y,MINBLOCK,true);
      PERKS_INITIALIZE_ALL_TYPE_5ARG(PERKS_DECLARE_INITIONIZATION_GENERALBOX,8,HALO,REG_FOLDER_Y,MINBLOCK,false);
      PERKS_INITIALIZE_ALL_TYPE_5ARG(PERKS_DECLARE_INITIONIZATION_GENERALBOX,16,HALO,REG_FOLDER_Y,MINBLOCK,true);
      PERKS_INITIALIZE_ALL_TYPE_5ARG(PERKS_DECLARE_INITIONIZATION_GENERALBOX,16,HALO,REG_FOLDER_Y,MINBLOCK,false);
  #endif
#else 
  #ifndef BOX
      PERKS_INITIALIZE_ALL_TYPE_5ARG(PERKS_DECLARE_INITIONIZATION_GENERAL_ASYNC,8,HALO,REG_FOLDER_Y,MINBLOCK,true);
      PERKS_INITIALIZE_ALL_TYPE_5ARG(PERKS_DECLARE_INITIONIZATION_GENERAL_ASYNC,8,HALO,REG_FOLDER_Y,MINBLOCK,false);
      PERKS_INITIALIZE_ALL_TYPE_5ARG(PERKS_DECLARE_INITIONIZATION_GENERAL_ASYNC,16,HALO,REG_FOLDER_Y,MINBLOCK,true);
      PERKS_INITIALIZE_ALL_TYPE_5ARG(PERKS_DECLARE_INITIONIZATION_GENERAL_ASYNC,16,HALO,REG_FOLDER_Y,MINBLOCK,false);
  #else
      PERKS_INITIALIZE_ALL_TYPE_5ARG(PERKS_DECLARE_INITIONIZATION_GENERALBOX_ASYNC,8,HALO,REG_FOLDER_Y,MINBLOCK,true);
      PERKS_INITIALIZE_ALL_TYPE_5ARG(PERKS_DECLARE_INITIONIZATION_GENERALBOX_ASYNC,8,HALO,REG_FOLDER_Y,MINBLOCK,false);
      PERKS_INITIALIZE_ALL_TYPE_5ARG(PERKS_DECLARE_INITIONIZATION_GENERALBOX_ASYNC,16,HALO,REG_FOLDER_Y,MINBLOCK,true);
      PERKS_INITIALIZE_ALL_TYPE_5ARG(PERKS_DECLARE_INITIONIZATION_GENERALBOX_ASYNC,16,HALO,REG_FOLDER_Y,MINBLOCK,false);
  #endif
#endif 
#else
  #ifdef USESM
    #define isUSESM true
  #else
    #define isUSESM false
  #endif
  template __global__ void 
#ifndef BOX
  #ifdef SMASYNC
    kernel_general_async
  #else 
    kernel_general
  #endif
#else
  #ifdef SMASYNC
    kernel_general_box_async
  #else 
    kernel_general_box
  #endif
#endif
  <TYPE,RTILE_Y,HALO,REG_FOLDER_Y,BLOCKTYPE,isUSESM>
  (TYPE*__restrict__,int,int,TYPE*__restrict__,TYPE*__restrict__,TYPE*__restrict__,int, int);
#endif