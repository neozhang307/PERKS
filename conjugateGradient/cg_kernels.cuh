#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include "cub/util_namespace.cuh"
#include "cub/agent/single_pass_scan_operators.cuh"
#include "cub/agent/agent_segment_fixup.cuh"
#include "cub/agent/agent_spmv_orig.cuh"
#include "cub/util_type.cuh"
#include "cub/util_debug.cuh"
#include "cub/util_device.cuh"

#include "cub/thread/thread_search.cuh"
#include "cub/grid/grid_queue.cuh"
#include "cub/config.cuh"
#include "cub/util_math.cuh"

#include "cub/util_type.cuh"
#include "cub/block/block_reduce.cuh"
#include "cub/block/block_scan.cuh"
#include "cub/block/block_exchange.cuh"
#include "cub/thread/thread_search.cuh"
#include "cub/thread/thread_operators.cuh"
#include "cub/iterator/cache_modified_input_iterator.cuh"
#include "cub/iterator/counting_input_iterator.cuh"


#include "cub/device/device_spmv.cuh"
#include "cub/util_allocator.cuh"
#include "cub/iterator/tex_ref_input_iterator.cuh"


#include "sparse_matrix.h"
#pragma once

template <
    typename        ValueT,              ///< Matrix and vector value type
    typename        tolValueT,              ///< Matrix and vector value type
    typename        OffsetT>             ///< Signed integer type for sequence offsets
struct CgParams
{
    ValueT*         d_val;            ///< Pointer to the array of \p num_nonzeros values of the corresponding nonzero elements of matrix <b>A</b>.
    OffsetT*        d_row_offsets;   ///< Pointer to the array of \p m offsets demarcating the end of every row in \p d_column_indices and \p d_values
    OffsetT*        d_column_indices;    ///< Pointer to the array of \p num_nonzeros column-indices of the corresponding nonzero elements of matrix <b>A</b>.  (Indices are zero-valued.)
    ValueT*         d_vector_x;          ///< Pointer to the array of \p num_cols values corresponding to the dense input vector <em>x</em>
    ValueT*         d_vector_Ax;          ///< Pointer to the array of \p num_rows values corresponding to the dense output vector <em>y</em>
    ValueT*         d_vector_p;          ///< Pointer to the array of \p num_rows values corresponding to the dense output vector <em>y</em>
    ValueT*         d_vector_r;          ///< Pointer to the array of \p num_rows values corresponding to the dense output vector <em>y</em>
    tolValueT*      d_vector_dot_results;          ///< Pointer to the array of \p num_rows values corresponding to the dense output vector <em>y</em>
    int             N;            ///< Number of rows of matrix <b>A</b>.
    int             nz;        ///< Number of nonzero elements of matrix <b>A</b>.
    ValueT          tol;               ///< Alpha multiplicand

    unsigned int*   d_iter;
    unsigned int    max_iter;
      unsigned int gridDim;
    // TexObjInputIterator<ValueT, OffsetT>  t_vector_x;
};
template<typename OffsetT,typename OffsetTLarge>  
struct SMCacheParams
{
    OffsetT sm_size_coor;
    OffsetT sm_size_total_coor;
    OffsetT sm_size_unit_coor;

    OffsetT sm_size_unit_matrix;
    OffsetT sm_size_blocktile_matrix;
    OffsetT sm_size_blocktile_total_matrix;
    OffsetT sm_num_matrixperblk;

    // OffsetT sm_size_thread_coor;
    // OffsetT sm_size_total_thread_coor;
    // OffsetT sm_size_unit_thread_coor;

    // OffsetT sm_size_vals;
    // OffsetT sm_size_unit_vals;
    // OffsetT sm_size_total_vals;

    // OffsetT sm_size_cols;
    // OffsetT sm_size_unit_cols;
    // OffsetT sm_size_total_cols;

    OffsetT sm_num_r;
    OffsetT sm_blk_size_r;
    // OffsetT sm_size_unit_x;
    // OffsetT sm_size_total_x;
    // OffsetT sm_size_unit_mat;
    OffsetTLarge sMemSize;
    OffsetTLarge sMemSizeTempt;
    OffsetTLarge sMemSizeTemptTotal;
};

namespace cg = cooperative_groups;


/// Optional outer namespace(s)
CUB_NS_PREFIX









/// CUB namespace
namespace cub {


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



template <typename        ValueT,
          typename        OffsetT>
__device__ __forceinline__ void gpuSaxpy_cub(ValueT *x, ValueT *y, ValueT a, OffsetT size,
                         const cg::grid_group &grid) {
  for (int i = grid.thread_rank(); i < size; i += grid.size()) {
    y[i] = a * x[i] + y[i];
  }
}


template <typename        ValueT,
          typename        OffsetT>
__device__ __forceinline__ void gpuDotProduct_cub(ValueT *vecA, ValueT *vecB, double *result,
                              OffsetT size, const cg::thread_block &cta,
                              double * tmp,
                              const cg::grid_group &grid) {
  // extern __shared__ double tmp[];

  double temp_sum = 0.0;
  for (int i = grid.thread_rank(); i < size; i += grid.size()) {
    temp_sum += static_cast<double>(vecA[i] * vecB[i]);
  }

  cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

  temp_sum = cg::reduce(tile32, temp_sum, cg::plus<double>());

  if (tile32.thread_rank() == 0) {
    tmp[tile32.meta_group_rank()] = temp_sum;
  }

//   cg::sync(cta);
  cta.sync();

  if (tile32.meta_group_rank() == 0) {
     temp_sum = tile32.thread_rank() < tile32.meta_group_size() ? tmp[tile32.thread_rank()] : 0.0;
     temp_sum = cg::reduce(tile32, temp_sum, cg::plus<double>());

    if (tile32.thread_rank() == 0) {
      atomicAdd(result, temp_sum);
    }
  }
}

template <typename        ValueT,
          typename        OffsetT>
__device__ __forceinline__ void gpuCopyVector_cub(ValueT *srcA, ValueT *destB, OffsetT size,
                              const cg::grid_group &grid) {
  for (int i = grid.thread_rank(); i < size; i += grid.size()) {
    destB[i] = srcA[i];
  }
}
template <typename        ValueT,
          typename        OffsetT>
__device__ __forceinline__ void gpuScaleVectorAndSaxpy_cub(const ValueT *x, ValueT *y, ValueT a, ValueT scale, OffsetT size,
                         const cg::grid_group &grid) {
  for (int i = grid.thread_rank(); i < size; i += grid.size()) {
    y[i] = a * x[i] + scale * y[i];
  }
}

/**
     * Spmv agent entry point
     */
template <
    typename        SpmvPolicyT,                ///< Parameterized SpmvPolicy tuning policy type
    typename        ScanTileStateT,             ///< Tile status interface type
    typename        ValueT,                     ///< Matrix and vector value type
    typename        OffsetT,                    ///< Signed integer type for sequence offsets
    typename        CoordinateT,                ///< Merge path coordinate type
    bool            HAS_ALPHA,                  ///< Whether the input parameter Alpha is 1
    bool            HAS_BETA>                   ///< Whether the input parameter Beta is 0
// __launch_bounds__ (int(SpmvPolicyT::BLOCK_THREADS))
__device__ void DvFuncSpmvKernel(
    SpmvParams<ValueT, OffsetT>     spmv_params,                ///< [in] SpMV input parameter bundle
    CoordinateT*                    d_tile_coordinates,         ///< [in] Pointer to the temporary array of tile starting coordinates
    KeyValuePair<OffsetT,ValueT>*   d_tile_carry_pairs,         ///< [out] Pointer to the temporary array carry-out dot product row-ids, one per block
    OffsetT                             num_merge_tiles,                  ///< [in] Number of merge tiles
    ScanTileStateT                  tile_state,                 ///< [in] Tile status interface for fixup reduce-by-key kernel
    OffsetT                             num_segment_fixup_tiles)    ///< [in] Number of reduce-by-key tiles (fixup grid size)
    // void *                          sm = NULL)
{
    // Spmv agent type specialization
    typedef AgentSpmv<
            SpmvPolicyT,
            ValueT,
            OffsetT,
            HAS_ALPHA, 
            HAS_BETA>
        AgentSpmvT;

    // Shared memory for AgentSpmv

    __shared__ typename AgentSpmvT::TempStorage temp_storage;
    // typename AgentSpmvT::IniTempStorage temp_storage(sm);
    // typename AgentSpmvT::_TempStorage *tmp;
    // (typename AgentSpmvT::TempStorage)sm;
    // typename AgentSpmvT::TempStorage temp_storage = &((AgentSpmvT::TempStorage*)sm)
    // int tile_idx=(blockIdx.x * gridDim.y) + blockIdx.y;
    for(int tile_idx=blockIdx.x; tile_idx<num_merge_tiles; tile_idx+=gridDim.x)
    {
        AgentSpmvT(temp_storage, spmv_params).ConsumeTile(
            tile_idx,
            d_tile_coordinates,
            d_tile_carry_pairs,
            num_merge_tiles);
    }

    // Initialize fixup tile status
    tile_state.InitializeStatus(num_segment_fixup_tiles);

}

template <
    typename        SpmvPolicyT,                ///< Parameterized SpmvPolicy tuning policy type
    typename        ScanTileStateT,             ///< Tile status interface type
    typename        ValueT,                     ///< Matrix and vector value type
    typename        OffsetT,                    ///< Signed integer type for sequence offsets
    typename        CoordinateT,                ///< Merge path coordinate type
    bool            HAS_ALPHA,                  ///< Whether the input parameter Alpha is 1
    bool            HAS_BETA>                   ///< Whether the input parameter Beta is 0
// __launch_bounds__ (int(SpmvPolicyT::BLOCK_THREADS))
__device__ void DvFuncSpmvKernelSM(
    SpmvParams<ValueT, OffsetT>     spmv_params,                ///< [in] SpMV input parameter bundle
    CoordinateT*                    d_tile_coordinates,         ///< [in] Pointer to the temporary array of tile starting coordinates
    CoordinateT*                    sm_tile_coordinates,         ///< [in] Pointer to the temporary array of tile starting coordinates
    KeyValuePair<OffsetT,ValueT>*   d_tile_carry_pairs,         ///< [out] Pointer to the temporary array carry-out dot product row-ids, one per block
    OffsetT                             num_merge_tiles,                  ///< [in] Number of merge tiles
    ScanTileStateT                  tile_state,                 ///< [in] Tile status interface for fixup reduce-by-key kernel
    OffsetT                             num_segment_fixup_tiles,    ///< [in] Number of reduce-by-key tiles (fixup grid size)
    bool                            loaded)
    // void *                          sm = NULL)
{
    // Spmv agent type specialization
    typedef AgentSpmv<
            SpmvPolicyT,
            ValueT,
            OffsetT,
            HAS_ALPHA,
            HAS_BETA>
        AgentSpmvT;

    // Shared memory for AgentSpmv

    __shared__ typename AgentSpmvT::TempStorage temp_storage;
    // typename AgentSpmvT::IniTempStorage temp_storage(sm);
    // typename AgentSpmvT::_TempStorage *tmp;
    // (typename AgentSpmvT::TempStorage)sm;
    // typename AgentSpmvT::TempStorage temp_storage = &((AgentSpmvT::TempStorage*)sm)
    // int tile_idx=(blockIdx.x * gridDim.y) + blockIdx.y;
    int sm_id=0;
    for(int tile_idx=blockIdx.x; tile_idx<num_merge_tiles; tile_idx+=gridDim.x)
    {
        AgentSpmvT(temp_storage, spmv_params).ConsumeTile(
            tile_idx,
            d_tile_coordinates,
            sm_tile_coordinates,
            d_tile_carry_pairs,
            num_merge_tiles,
            sm_id++
            ,
            loaded
            );
    }

    // Initialize fixup tile status
    tile_state.InitializeStatus(num_segment_fixup_tiles);

}


template <
    typename        SpmvPolicyT,                ///< Parameterized SpmvPolicy tuning policy type
    typename        ScanTileStateT,             ///< Tile status interface type
    typename        ValueT,                     ///< Matrix and vector value type
    typename        OffsetT,                    ///< Signed integer type for sequence offsets
    typename        CoordinateT,                ///< Merge path coordinate type
    bool            HAS_ALPHA,                  ///< Whether the input parameter Alpha is 1
    bool            HAS_BETA>                   ///< Whether the input parameter Beta is 0
// __launch_bounds__ (int(SpmvPolicyT::BLOCK_THREADS))
__device__ void DvFuncSpmvKernelSM2(
    SpmvParams<ValueT, OffsetT>     spmv_params,                ///< [in] SpMV input parameter bundle
    CoordinateT*                    d_tile_coordinates,         ///< [in] Pointer to the temporary array of tile starting coordinates
    CoordinateT*                    sm_tile_coordinates,         ///< [in] Pointer to the temporary array of tile starting coordinates
    CoordinateT*                    sm_thread_tile_coordinates,         ///< [in] Pointer to the temporary array of tile starting coordinates
    KeyValuePair<OffsetT,ValueT>*   d_tile_carry_pairs,         ///< [out] Pointer to the temporary array carry-out dot product row-ids, one per block
    OffsetT                             num_merge_tiles,                  ///< [in] Number of merge tiles
    ScanTileStateT                  tile_state,                 ///< [in] Tile status interface for fixup reduce-by-key kernel
    OffsetT                             num_segment_fixup_tiles,    ///< [in] Number of reduce-by-key tiles (fixup grid size)
    bool                            loaded,
    OffsetT                         sm_size_unit_thread_coor)
    // void *                          sm = NULL)
{
    // Spmv agent type specialization
    typedef AgentSpmv<
            SpmvPolicyT,
            ValueT,
            OffsetT,
            HAS_ALPHA,
            HAS_BETA>
        AgentSpmvT;
    
    // Shared memory for AgentSpmv

    __shared__ typename AgentSpmvT::TempStorage temp_storage;
    // typename AgentSpmvT::IniTempStorage temp_storage(sm);
    // typename AgentSpmvT::_TempStorage *tmp;
    // (typename AgentSpmvT::TempStorage)sm;
    // typename AgentSpmvT::TempStorage temp_storage = &((AgentSpmvT::TempStorage*)sm)
    // int tile_idx=(blockIdx.x * gridDim.y) + blockIdx.y;
    int sm_id=0;
    for(int tile_idx=blockIdx.x; tile_idx<num_merge_tiles; tile_idx+=gridDim.x)
    {
        if(sm_id<sm_size_unit_thread_coor)//+sm_id*blockDim.x
        // if(false)
        {
            AgentSpmvT(temp_storage, spmv_params).ConsumeTile(
                tile_idx,
                d_tile_coordinates,
                sm_tile_coordinates,
                sm_thread_tile_coordinates+sm_id*blockDim.x,
                d_tile_carry_pairs,
                num_merge_tiles,
                sm_id++,
                loaded);
        }
        else
        {
            AgentSpmvT(temp_storage, spmv_params).ConsumeTile(
                tile_idx,
                d_tile_coordinates,
                sm_tile_coordinates,
                // sm_thread_tile_coordinates+sm_id*blockDim.x,
                d_tile_carry_pairs,
                num_merge_tiles,
                sm_id++,
                loaded);
        }
    }

    // Initialize fixup tile status
    tile_state.InitializeStatus(num_segment_fixup_tiles);

}

template <
    typename    AgentSegmentFixupPolicyT,       ///< Parameterized AgentSegmentFixupPolicy tuning policy type
    typename    PairsInputIteratorT,            ///< Random-access input iterator type for keys
    typename    AggregatesOutputIteratorT,      ///< Random-access output iterator type for values
    typename    OffsetT,                        ///< Signed integer type for global offsets
    typename    ScanTileStateT>                 ///< Tile status interface type
// __launch_bounds__ (int(AgentSegmentFixupPolicyT::BLOCK_THREADS))
__device__ void DvFuncSegmentFixupKernel(
    PairsInputIteratorT         d_tile_carry_pairs,         ///< [in] Pointer to the array carry-out dot product row-ids, one per spmv block
    AggregatesOutputIteratorT   d_aggregates_out,   ///< [in,out] Output value aggregates
    OffsetT                     num_merge_tiles,          ///< [in] Total number of items to select from
    int                         num_segment_fixup_tiles,          ///< [in] Total number of tiles for the entire problem
    // int                         range,
    ScanTileStateT              tile_state)         ///< [in] Tile status interface
    // void *                      sm)
{
    // Thread block type for reducing tiles of value segments
    typedef AgentSegmentFixup<
            AgentSegmentFixupPolicyT,
            PairsInputIteratorT,
            AggregatesOutputIteratorT,
            cub::Equality,
            cub::Sum,
            OffsetT>
        AgentSegmentFixupT;

    // Shared memory for AgentSegmentFixup
    __shared__ typename AgentSegmentFixupT::TempStorage temp_storage;
    for(int tile_idx=blockIdx.x; tile_idx<num_segment_fixup_tiles; tile_idx+=gridDim.x)
    {
        // Process tiles
        AgentSegmentFixupT(temp_storage, d_tile_carry_pairs, d_aggregates_out, cub::Equality(), cub::Sum()).ConsumeRange(
            num_merge_tiles,
            num_segment_fixup_tiles,
            tile_state,
            tile_idx);
        //  AgentSegmentFixupT(temp_storage, d_tile_carry_pairs, d_aggregates_out, cub::Equality(), cub::Sum()).ConsumeRange(
        //     tile_idx,  
        //     num_merge_tiles,
        //     num_segment_fixup_tiles,
        //     tile_state);
    }
}


template <
    typename        SpmvPolicyT,                ///< Parameterized SpmvPolicy tuning policy type
    typename        ScanTileStateT,             ///< Tile status interface type
    typename        ValueT,                     ///< Matrix and vector value type
    typename        OffsetT,                    ///< Signed integer type for sequence offsets
    typename        CoordinateT,                ///< Merge path coordinate type
    bool            HAS_ALPHA,                  ///< Whether the input parameter Alpha is 1
    bool            HAS_BETA,//>                   ///< Whether the input parameter Beta is 0
    // <
    typename    AgentSegmentFixupPolicyT,       ///< Parameterized AgentSegmentFixupPolicy tuning policy type
    // typename    PairsInputIteratorT,            ///< Random-access input iterator type for keys
    typename    AggregatesOutputIteratorT,
    bool isBaseline=true,
    bool cacheMatrix=false,
    bool cacheVector=false,
    bool isStaticIteration=false
    >//,      ///< Random-access output iterator type for values
    // typename    OffsetT,                        ///< Signed integer type for global offsets
            // typename    ScanTileStateT>                 ///< Tile status interface type
// __launch_bounds__(256,2)
__launch_bounds__(128,5)
__global__ void gpuConjugateGradient_cub
(
    SpmvParams<ValueT, OffsetT>     spmv_params,                ///< [in] SpMV input parameter bundle
    CoordinateT*                    d_tile_coordinates,         ///< [in] Pointer to the temporary array of tile starting coordinates
    KeyValuePair<OffsetT,ValueT>*   d_tile_carry_pairs,         ///< [out] Pointer to the temporary array carry-out dot product row-ids, one per block
    int                             num_merge_tiles,                  ///< [in] Number of merge tiles
    ScanTileStateT                  tile_state,                 ///< [in] Tile status interface for fixup reduce-by-key kernel
    int                             num_segment_fixup_tiles,    ///< [in] Number of reduce-by-key tiles (fixup grid size)
    // int                             range_segment,        
    // AggregatesOutputIteratorT   d_aggregates_out,   ///< [in,out] Output value aggregates
    OffsetT *I, OffsetT *J, ValueT *val,
    ValueT *x, ValueT *Ax, ValueT *p,
    ValueT *r, double *dot_result,
    OffsetT nnz, OffsetT N, ValueT tol,
    SMCacheParams<OffsetT,size_t> smParamsT,
    unsigned int * iteration,
    unsigned int maxiter,
    OffsetT workCountSMX
) {

    cg::thread_block cta = cg::this_thread_block();
    cg::grid_group grid = cg::this_grid();
    extern __shared__ char tmp_ori[];
    double *tmp=(double*)tmp_ori;

    CoordinateT *sm_tile_coordinates=(CoordinateT *)(tmp_ori+smParamsT.sMemSizeTempt);
    // CoordinateT *sm_tile_coordinates=(CoordinateT *)(tmp_ori+smParamsT.sMemSizeTemptTotal);
    // CoordinateT *sm_tile_thread_coordinates=sm_tile_coordinates+smParamsT.sm_size_unit_coor;
    // __shared__ CoordinateT *sm_tile
    // auto spmv_kernel=DvFuncSpmvKernelSM2<SpmvPolicyT, ScanTileStateT, ValueT, OffsetT, CoordinateT, true, false>;
    // auto segment_fixup_kernel=DvFuncSegmentFixupKernel<AgentSegmentFixupPolicyT, KeyValuePair<OffsetT,ValueT>*, ValueT*, OffsetT, ScanTileStateT>;

    unsigned int max_iter = maxiter;//N;//MAXITER;

    ValueT alpha = 1.0;
    ValueT alpham1 = -1.0;
    ValueT r0 = 0.0, r1, b, a, na;

    // gpuSpMV_cub(I, J, val, nnz, N, alpha, x, Ax, cta, grid);
    spmv_params.d_vector_x=x;

    typedef AgentSpmv<
                SpmvPolicyT,
                ValueT,
                OffsetT,
                true,
                false>
            AgentSpmvT;
    // __shared__ typename AgentSpmvT::TempStorage temp_storage;
    typename AgentSpmvT::TempStorage* temp_storage=((typename AgentSpmvT::TempStorage*)tmp_ori);
    typedef AgentSegmentFixup<
                AgentSegmentFixupPolicyT,
                KeyValuePair<OffsetT,ValueT>*,
                ValueT *,
                cub::Equality,
                cub::Sum,
                OffsetT>
            AgentSegmentFixupT;

        // Shared memory for AgentSegmentFixup
    // __shared__ typename AgentSegmentFixupT::TempStorage temp_storage2;
    typename AgentSegmentFixupT::TempStorage* temp_storage2=((typename AgentSegmentFixupT::TempStorage*)tmp);

    typename AgentSpmvT::MatrixTileUnit* matrixunits 
            = 
            (typename AgentSpmvT::MatrixTileUnit*)(tmp_ori+smParamsT.sm_size_coor+smParamsT.sMemSizeTempt);
    
    ValueT * sm_r = (ValueT*)(matrixunits+smParamsT.sm_num_matrixperblk); 
    const unsigned int g_start =grid.thread_rank(); 
    const unsigned int sm_num_r=smParamsT.sm_num_r;
    const unsigned int g_sm_r_middle= g_start+sm_num_r*grid.size();
    const int sm_r_start = cta.thread_rank();//grid.thread_rank(); 
    const int g_step  = grid.size();//grid.size();
    const int b_step  = cta.size();
    // for(int i=0; i<sm_num_r; i++)
    // {
    //     sm_r[i*blockDim.x+threadIdx.x]=sm_r_start;
    // }


    {
        if(isBaseline)
        {
            for(int tile_idx=blockIdx.x; tile_idx<num_merge_tiles; tile_idx+=gridDim.x)
            {
                AgentSpmvT(temp_storage[0], spmv_params).ConsumeTilePERKS(
                    // tile_idx,
                    d_tile_coordinates,
                    d_tile_carry_pairs,
                    num_merge_tiles,
                    tile_idx);
            }
            // assert(-1);
            // return;
        }
        else
        {
            int sm_id=0;
            for(int tile_idx=blockIdx.x; tile_idx<num_merge_tiles; tile_idx+=gridDim.x)
            {
                AgentSpmvT(temp_storage[0], spmv_params).ConsumeTilePERKS
                <false,cacheMatrix>(
                    d_tile_coordinates,
                    sm_tile_coordinates,
                    d_tile_carry_pairs,
                    num_merge_tiles,
                    sm_id++,
                    tile_idx,
                    matrixunits,
                    smParamsT.sm_num_matrixperblk);
            }
            // if(threadIdx.x==0)
            // {
            //     printf("<%d,%d,%d,%d,%d>",sm_id,smParamsT.sm_size_unit_coor,num_merge_tiles/gridDim.x,num_merge_tiles,gridDim.x);
            // }
            // return;
            // assert(false);
            // return;
        }

        // Initialize fixup tile status
        tile_state.InitializeStatus(num_segment_fixup_tiles);
    }

    cg::sync(grid);

    if (num_merge_tiles > 1)
    {
        for(int tile_idx=blockIdx.x; tile_idx<num_segment_fixup_tiles; tile_idx+=gridDim.x)
        {
            // Process tiles
            AgentSegmentFixupT(temp_storage2[0], d_tile_carry_pairs, Ax, cub::Equality(), cub::Sum()).ConsumeRange(
                num_merge_tiles,
                num_segment_fixup_tiles,
                tile_state,
                tile_idx);
        }
    }

    cg::sync(grid);
    spmv_params.d_vector_x=p;

    // 
    if(cacheVector)
    {
        gpuCopyVector_perk_reg(
        r, g_start, g_step, 
        sm_r,  sm_r_start, b_step,  sm_num_r);
        
        cg::sync(grid);
        
        gpuSaxpy_perk_reg(Ax, g_start, g_step,
        sm_r,  sm_r_start, b_step,  sm_num_r,
        alpham1); 

        gpuSaxpy_perk<ValueT,OffsetT> (Ax, //g_sm_middle, g_end, g_step,
                r,   g_sm_r_middle, N, grid.size(),
                alpham1);
    }
    else
    {
        gpuSaxpy_cub(Ax, r, alpham1, N, grid);
    }
    // cta.sync();


    cg::sync(grid);

    // 
    if(cacheVector)
    {
        double temp_sum=0.0;
        gpuDotProduct_perk_pre_reg(sm_r,  sm_r_start,b_step,
                            sm_r,  sm_r_start, b_step, sm_num_r ,
                            temp_sum);
        gpuDotProduct_perk_pre<ValueT,OffsetT>(r, // g_sm_middle, g_end, g_step,
                                r,   g_sm_r_middle, N, g_step,
                                temp_sum);


        gpuDotProduct_perk_post(dot_result, temp_sum, cta,tmp,grid);

    }
    else
    {
        gpuDotProduct_cub(r, r, dot_result, N, cta, tmp, grid);
    }
    cg::sync(grid);

    r1 = *dot_result;

    int k = 1;
    while (true) {
      if((!isStaticIteration&r1 < tol * tol) || k > max_iter)break;
      // if(k > max_iter)break;
    // while (k <= max_iter) {
      // printf("%d");
            if (k > 1) {
                b = r1 / r0;
                //v(r)+p(p)->v(p)
                // 
                if(cacheVector)
                {
                    gpuScaleVectorAndSaxpy_perk_reg(sm_r,  sm_r_start, b_step,  
                                        p,  g_start, g_step, sm_num_r,
                                        alpha, b);
                    gpuScaleVectorAndSaxpy_perk<ValueT,OffsetT>(r,  
                                            p,   g_sm_r_middle, N, grid.size(),
                                            alpha, b);
                }
                else
                {
                    gpuScaleVectorAndSaxpy_cub(r, p, alpha, b, N, grid);
                }
                // gpuCopyVector_cub(r, p, N, grid);
            } else {
            //v(r)->v(p)
            // 
            if(cacheVector)
            {
                gpuCopyVector_perk_reg(sm_r,  sm_r_start, b_step, 
                      p,  g_start, g_step, sm_num_r);
                gpuCopyVector_perk<ValueT,OffsetT>(r,
                                p,   g_sm_r_middle, N, g_step);
            }
            else
            {
                gpuCopyVector_cub(r, p, N, grid);
            }
        }
        if (threadIdx.x == 0 && blockIdx.x == 0){ dot_result[1] = 0.0;dot_result[0]=0.0;}
        cg::sync(grid);
        //max(IJv) + v(p) -> v(Ax) 

        {
            if(isBaseline)
            {
                for(int tile_idx=blockIdx.x; tile_idx<num_merge_tiles; tile_idx+=gridDim.x)
                {
                    AgentSpmvT(temp_storage[0], spmv_params).ConsumeTilePERKS(
                        // tile_idx,
                        d_tile_coordinates,
                        d_tile_carry_pairs,
                        num_merge_tiles,
                        tile_idx);
                }
            }
            else
            {
                int sm_id=0;
                for(int tile_idx=blockIdx.x; tile_idx<num_merge_tiles; tile_idx+=gridDim.x)
                {
                    AgentSpmvT(temp_storage[0], spmv_params).ConsumeTilePERKS
                    <true,cacheMatrix>(
                        d_tile_coordinates,
                        sm_tile_coordinates,
                        d_tile_carry_pairs,
                        num_merge_tiles,
                        sm_id++,
                        tile_idx,
                        matrixunits,
                        smParamsT.sm_num_matrixperblk
                        );     
                }
            }

            // Initialize fixup tile status
            tile_state.InitializeStatus(num_segment_fixup_tiles);
        } 
        cg::sync(grid);
        if (num_merge_tiles > 1)
        {
            for(int tile_idx=blockIdx.x; tile_idx<num_segment_fixup_tiles; tile_idx+=gridDim.x)
            {
                // Process tiles
                AgentSegmentFixupT(temp_storage2[0], d_tile_carry_pairs, Ax, cub::Equality(), cub::Sum()).ConsumeRange(
                    num_merge_tiles,
                    num_segment_fixup_tiles,
                    tile_state,
                    tile_idx);
            }
        }
        // Shared memory for AgentSegmentFixup
        cg::sync(grid);
        //v(p)+v(Ax) -> v(dot_result)
        gpuDotProduct_cub(p, Ax, (dot_result+1), N, cta, tmp, grid);

        cg::sync(grid);

        a = r1 / *(dot_result+1);
        //v(p)+v(x)->v(x)
        gpuSaxpy_cub(p, x, a, N, grid);
        na = -a;
        //v(Ax)+v(r)->v(na)
        // 
        if(cacheVector)
        {
            gpuSaxpy_perk_reg(Ax,  g_start, g_step,
                    sm_r, sm_r_start, b_step, sm_num_r,
                    na);

            gpuSaxpy_perk<ValueT,OffsetT>(Ax,  //g_sm_middle, g_end, g_step,
                        r,  g_sm_r_middle, N, grid.size(),
                        na);
        }
        else
        {
            gpuSaxpy_cub(Ax, r, na, N, grid);
        }
        // if(threadIdx.x==0&&blockIdx.x==0)printf("<%f,%f>",);
        r0 = r1;
        // 
        if(cacheVector)
        {
            double temp_sum=0.0;
            gpuDotProduct_perk_pre_reg(sm_r,  sm_r_start,  b_step, 
                                sm_r,  sm_r_start,  b_step, sm_num_r,
                                temp_sum);

            gpuDotProduct_perk_pre<ValueT,OffsetT>(r, // g_sm_middle, g_end, g_step,
                                    r,   g_sm_r_middle, N, g_step,
                                    temp_sum);
            
            gpuDotProduct_perk_post(dot_result, temp_sum, cta,tmp,grid);
        }
        else
        {
            gpuDotProduct_cub(r, r, (dot_result), N, cta, tmp, grid);
        }
        cg::sync(grid);

        r1 = *(dot_result);
        k++;
        // if(threadIdx.x==0&&blockIdx.x==0)printf("<%e>",r1);
    }
    // gpuSaxpy_cub(p, x, a, N, grid);
    if(threadIdx.x==0&&blockIdx.x==0)iteration[0]=k-1;
    if(cacheVector)
    {
        gpuCopyVector_perk_reg(
            sm_r,  sm_r_start, b_step,  
            r, g_start, g_step, 
            sm_num_r);
    }

}

    /******************************************************************************
     * SpMV kernel entry points
     *****************************************************************************/

/**
  * Spmv search kernel. Identifies merge path starting coordinates for each tile.
  */
template <
    typename    AgentSpmvPolicyT,           ///< Parameterized SpmvPolicy tuning policy type
    typename    ValueT,                     ///< Matrix and vector value type
    typename    OffsetT>                    ///< Signed integer type for sequence offsets
__global__ void MyDeviceSpmv1ColKernel(
    SpmvParams<ValueT, OffsetT> spmv_params)                ///< [in] SpMV input parameter bundle
{
    typedef CacheModifiedInputIterator<
            AgentSpmvPolicyT::VECTOR_VALUES_LOAD_MODIFIER,
            ValueT,
            OffsetT>
        VectorValueIteratorT;

    VectorValueIteratorT wrapped_vector_x(spmv_params.d_vector_x);

    int row_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (row_idx < spmv_params.num_rows)
    {
        OffsetT     end_nonzero_idx = spmv_params.d_row_end_offsets[row_idx];
        OffsetT     nonzero_idx = spmv_params.d_row_end_offsets[row_idx - 1];

        ValueT value = 0.0;
        if (end_nonzero_idx != nonzero_idx)
        {
            value = spmv_params.d_values[nonzero_idx] * wrapped_vector_x[spmv_params.d_column_indices[nonzero_idx]];
        }

        spmv_params.d_vector_y[row_idx] = value;
    }
}


/**
  * Spmv search kernel. Identifies merge path starting coordinates for each tile.
  */
template <
    typename    SpmvPolicyT,                    ///< Parameterized SpmvPolicy tuning policy type
    typename    OffsetT,                        ///< Signed integer type for sequence offsets
    typename    CoordinateT,                    ///< Merge path coordinate type
    typename    SpmvParamsT>                    ///< SpmvParams type
__global__ void MyDeviceSpmvSearchKernel(
    int             num_merge_tiles,            ///< [in] Number of SpMV merge tiles (spmv grid size)
    CoordinateT*    d_tile_coordinates,         ///< [out] Pointer to the temporary array of tile starting coordinates
    SpmvParamsT     spmv_params)                ///< [in] SpMV input parameter bundle
{
    /// Constants
    enum
    {
        BLOCK_THREADS           = SpmvPolicyT::BLOCK_THREADS,
        ITEMS_PER_THREAD        = SpmvPolicyT::ITEMS_PER_THREAD,
        TILE_ITEMS              = BLOCK_THREADS * ITEMS_PER_THREAD,
    };

    typedef CacheModifiedInputIterator<
            SpmvPolicyT::ROW_OFFSETS_SEARCH_LOAD_MODIFIER,
            OffsetT,
            OffsetT>
        RowOffsetsSearchIteratorT;

    // Find the starting coordinate for all tiles (plus the end coordinate of the last one)
    int tile_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tile_idx < num_merge_tiles + 1)
    {
        OffsetT                         diagonal = (tile_idx * TILE_ITEMS);
        CoordinateT                     tile_coordinate;
        CountingInputIterator<OffsetT>  nonzero_indices(0);

        // Search the merge path
        MergePathSearch(
            diagonal,
            RowOffsetsSearchIteratorT(spmv_params.d_row_end_offsets),
            nonzero_indices,
            spmv_params.num_rows,
            spmv_params.num_nonzeros,
            tile_coordinate);

        // Output starting offset
        d_tile_coordinates[tile_idx] = tile_coordinate;
    }
}


/**
  * Spmv agent entry point
  */
template <
    typename        SpmvPolicyT,                ///< Parameterized SpmvPolicy tuning policy type
    typename        ScanTileStateT,             ///< Tile status interface type
    typename        ValueT,                     ///< Matrix and vector value type
    typename        OffsetT,                    ///< Signed integer type for sequence offsets
    typename        CoordinateT,                ///< Merge path coordinate type
    bool            HAS_ALPHA,                  ///< Whether the input parameter Alpha is 1
    bool            HAS_BETA>                   ///< Whether the input parameter Beta is 0
__launch_bounds__ (int(SpmvPolicyT::BLOCK_THREADS))
__global__ void MyDeviceSpmvKernel(
    SpmvParams<ValueT, OffsetT>     spmv_params,                ///< [in] SpMV input parameter bundle
    CoordinateT*                    d_tile_coordinates,         ///< [in] Pointer to the temporary array of tile starting coordinates
    KeyValuePair<OffsetT,ValueT>*   d_tile_carry_pairs,         ///< [out] Pointer to the temporary array carry-out dot product row-ids, one per block
    int                             num_merge_tiles,                  ///< [in] Number of merge tiles
    ScanTileStateT                  tile_state,                 ///< [in] Tile status interface for fixup reduce-by-key kernel
    int                             num_segment_fixup_tiles)    ///< [in] Number of reduce-by-key tiles (fixup grid size)
{
    // Spmv agent type specialization
    typedef AgentSpmv<
            SpmvPolicyT,
            ValueT,
            OffsetT,
            HAS_ALPHA,
            HAS_BETA>
        AgentSpmvT;

    // Shared memory for AgentSpmv
    __shared__ typename AgentSpmvT::TempStorage temp_storage;

    // int tile_idx=(blockIdx.x * gridDim.y) + blockIdx.y;
    for(int tile_idx=blockIdx.x; tile_idx<num_merge_tiles; tile_idx+=gridDim.x)
    {
        // AgentSpmvT(temp_storage, spmv_params).ConsumeTile(
        //     tile_idx,  
        //     d_tile_coordinates,
        //     d_tile_carry_pairs,
        //     num_merge_tiles);
        AgentSpmvT(temp_storage, spmv_params).ConsumeTile(
            d_tile_coordinates,
            d_tile_carry_pairs,
            num_merge_tiles,
            tile_idx);
    }

    // Initialize fixup tile status
    tile_state.InitializeStatus(num_segment_fixup_tiles);

}

template <
    typename        SpmvPolicyT,                ///< Parameterized SpmvPolicy tuning policy type
    typename        ScanTileStateT,             ///< Tile status interface type
    typename        ValueT,                     ///< Matrix and vector value type
    typename        OffsetT,                    ///< Signed integer type for sequence offsets
    typename        CoordinateT,                ///< Merge path coordinate type
    bool            HAS_ALPHA,                  ///< Whether the input parameter Alpha is 1
    bool            HAS_BETA>                   ///< Whether the input parameter Beta is 0
__launch_bounds__ (int(SpmvPolicyT::BLOCK_THREADS))
__global__ void MyDeviceSpmvKernel_Origin(
    SpmvParams<ValueT, OffsetT>     spmv_params,                ///< [in] SpMV input parameter bundle
    CoordinateT*                    d_tile_coordinates,         ///< [in] Pointer to the temporary array of tile starting coordinates
    KeyValuePair<OffsetT,ValueT>*   d_tile_carry_pairs,         ///< [out] Pointer to the temporary array carry-out dot product row-ids, one per block
    int                             num_tiles,                  ///< [in] Number of merge tiles
    ScanTileStateT                  tile_state,                 ///< [in] Tile status interface for fixup reduce-by-key kernel
    int                             num_segment_fixup_tiles)    ///< [in] Number of reduce-by-key tiles (fixup grid size)
{
    // Spmv agent type specialization
    typedef AgentSpmv<
            SpmvPolicyT,
            ValueT,
            OffsetT,
            HAS_ALPHA,
            HAS_BETA>
        AgentSpmvT;

    // Shared memory for AgentSpmv
    __shared__ typename AgentSpmvT::TempStorage temp_storage;

    int tile_idx=(blockIdx.x * gridDim.y) + blockIdx.y;
    // for(int tile_idx=blockIdx.x; tile_idx<num_tiles; tile_idx+=gridDim.x)
    {
        AgentSpmvT(temp_storage, spmv_params).ConsumeTile(
            tile_idx,
            d_tile_coordinates,
            d_tile_carry_pairs,
            num_tiles);
    }

    // Initialize fixup tile status
    tile_state.InitializeStatus(num_segment_fixup_tiles);

}
    /**
     * Multi-block reduce-by-key sweep kernel entry point
     */
template <
    typename    AgentSegmentFixupPolicyT,       ///< Parameterized AgentSegmentFixupPolicy tuning policy type
    typename    PairsInputIteratorT,            ///< Random-access input iterator type for keys
    typename    AggregatesOutputIteratorT,      ///< Random-access output iterator type for values
    typename    OffsetT,                        ///< Signed integer type for global offsets
    typename    ScanTileStateT>                 ///< Tile status interface type
__launch_bounds__ (int(AgentSegmentFixupPolicyT::BLOCK_THREADS))
__global__ void MyDeviceSegmentFixupKernel(
    PairsInputIteratorT         d_tile_carry_pairs,         ///< [in] Pointer to the array carry-out dot product row-ids, one per spmv block
    AggregatesOutputIteratorT   d_aggregates_out,   ///< [in,out] Output value aggregates
    OffsetT                     num_merge_tiles,          ///< [in] Total number of items to select from
    int                         num_segment_fixup_tiles,          ///< [in] Total number of tiles for the entire problem
    ScanTileStateT              tile_state)         ///< [in] Tile status interface
{
    // Thread block type for reducing tiles of value segments
    typedef AgentSegmentFixup<
            AgentSegmentFixupPolicyT,
            PairsInputIteratorT,
            AggregatesOutputIteratorT,
            cub::Equality,
            cub::Sum,
            OffsetT>
        AgentSegmentFixupT;

    // Shared memory for AgentSegmentFixup
    __shared__ typename AgentSegmentFixupT::TempStorage temp_storage;
    for(int tile_idx=blockIdx.x; tile_idx<num_segment_fixup_tiles; tile_idx+=gridDim.x)
    {
        // Process tiles
        AgentSegmentFixupT(temp_storage, d_tile_carry_pairs, d_aggregates_out, cub::Equality(), cub::Sum()).ConsumeRange(
            num_merge_tiles,
            num_segment_fixup_tiles,
            tile_state,
            tile_idx);

        // AgentSegmentFixupT(temp_storage, d_tile_carry_pairs, d_aggregates_out, cub::Equality(), cub::Sum()).ConsumeRange(
        //     tile_idx,  
        //     num_merge_tiles,
        //     num_segment_fixup_tiles,
        //     tile_state);
    }
}


template <
    typename    AgentSegmentFixupPolicyT,       ///< Parameterized AgentSegmentFixupPolicy tuning policy type
    typename    PairsInputIteratorT,            ///< Random-access input iterator type for keys
    typename    AggregatesOutputIteratorT,      ///< Random-access output iterator type for values
    typename    OffsetT,                        ///< Signed integer type for global offsets
    typename    ScanTileStateT>                 ///< Tile status interface type
__launch_bounds__ (int(AgentSegmentFixupPolicyT::BLOCK_THREADS))
__global__ void MyDeviceSegmentFixupKernel_Origin(
    PairsInputIteratorT         d_pairs_in,         ///< [in] Pointer to the array carry-out dot product row-ids, one per spmv block
    AggregatesOutputIteratorT   d_aggregates_out,   ///< [in,out] Output value aggregates
    OffsetT                     num_items,          ///< [in] Total number of items to select from
    int                         num_tiles,          ///< [in] Total number of tiles for the entire problem
    ScanTileStateT              tile_state)         ///< [in] Tile status interface
{
    // Thread block type for reducing tiles of value segments
    typedef AgentSegmentFixup<
            AgentSegmentFixupPolicyT,
            PairsInputIteratorT,
            AggregatesOutputIteratorT,
            cub::Equality,
            cub::Sum,
            OffsetT>
        AgentSegmentFixupT;

    // Shared memory for AgentSegmentFixup
    __shared__ typename AgentSegmentFixupT::TempStorage temp_storage;

    // Process tiles
    AgentSegmentFixupT(temp_storage, d_pairs_in, d_aggregates_out, cub::Equality(), cub::Sum()).ConsumeRange(
        // (blockIdx.x * gridDim.y) + blockIdx.y,
        num_items,
        num_tiles,
        tile_state);
}

// CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)
