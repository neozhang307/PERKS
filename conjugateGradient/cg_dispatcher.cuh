

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
// #define THRUST_IGNORE_CUB_VERSION_CHECK

#pragma once
// #include <stdio.h>
// #include <map>
// #include <vector>
// #include <algorithm>
// #include <cstdio>
// #include <fstream>
// includes, system
// #include <stdio.h>
// #include <stdlib.h>
// #include <string.h>

#include "cg_kernels.cuh"
#include <cuda_runtime.h>

// Utilities and system includes
#include <helper_cuda.h>  // helper function CUDA error checking and initialization
#include <helper_functions.h>  // helper for shared functions common to CUDA Samples



// #include <type_traits>

#include "sparse_matrix.h"

#include "cub/device/device_spmv.cuh"

// #define THREADS_PER_BLOCK 256
#define THREADS_PER_BLOCK 128
// 
// #define THREADS_PER_BLOCK 128
// #define REG (3)
#define MAXITER (10000)
// #define BASELINE
// #ifndef BASELINE
// #define NOCOO
// #endif
// #define VEC
#define USEVDATA
// #define USEVDATA
// #define NOCOO
// #define VECR
// #define VECX
// #define VEC
// #define COO
// #define VAL
// #define COL
// #include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

CUB_NS_PREFIX


/******************************************************************************
    * Dispatch
    ******************************************************************************/

/**
    * Utility class for dispatching the appropriately-tuned kernels for DeviceSpmv
*/
template <
    typename    ValueT,                     ///< Matrix and vector value type
    typename    OffsetT,                    ///< Signed integer type for global offsets
    bool isbaseline=true,
    bool cacheMatrix=false,
    bool cacheVector=false>
struct DispatchCG
{
        //---------------------------------------------------------------------
        // Constants and Types
        //---------------------------------------------------------------------

    enum
    {
        INIT_KERNEL_THREADS = THREADS_PER_BLOCK
    };

    // SpmvParams bundle type
    typedef SpmvParams<ValueT, OffsetT> SpmvParamsT;

    // 2D merge path coordinate type
    typedef typename CubVector<OffsetT, 2>::Type CoordinateT;

    // Tile status descriptor interface type
    typedef ReduceByKeyScanTileState<ValueT, OffsetT> ScanTileStateT;

    // Tuple type for scanning (pairs accumulated segment-value with segment-index)
    typedef KeyValuePair<OffsetT, ValueT> KeyValuePairT;


        //---------------------------------------------------------------------
        // Tuning policies
        //---------------------------------------------------------------------

        // /// SM35
        // struct Policy350
        // {
        //     typedef AgentSpmvPolicy<
        //             (sizeof(ValueT) > 4) ? 96 : 128,
        //             (sizeof(ValueT) > 4) ? 4 : 7,
        //             LOAD_LDG,
        //             LOAD_CA,
        //             LOAD_LDG,
        //             LOAD_LDG,
        //             LOAD_LDG,
        //             (sizeof(ValueT) > 4) ? true : false,
        //             BLOCK_SCAN_WARP_SCANS>
        //         SpmvPolicyT;

        //     typedef AgentSegmentFixupPolicy<
        //             128,
        //             3,
        //             BLOCK_LOAD_VECTORIZE,
        //             LOAD_LDG,
        //             BLOCK_SCAN_WARP_SCANS>
        //         SegmentFixupPolicyT;
        // };


        // /// SM37
        // struct Policy370
        // {

        //     typedef AgentSpmvPolicy<
        //             (sizeof(ValueT) > 4) ? 128 : 128,
        //             (sizeof(ValueT) > 4) ? 9 : 14,
        //             LOAD_LDG,
        //             LOAD_CA,
        //             LOAD_LDG,
        //             LOAD_LDG,
        //             LOAD_LDG,
        //             false,
        //             BLOCK_SCAN_WARP_SCANS>
        //         SpmvPolicyT;

        //     typedef AgentSegmentFixupPolicy<
        //             128,
        //             3,
        //             BLOCK_LOAD_VECTORIZE,
        //             LOAD_LDG,
        //             BLOCK_SCAN_WARP_SCANS>
        //         SegmentFixupPolicyT;
        // };

        // /// SM50
        // struct Policy500
        // {
        //     typedef AgentSpmvPolicy<
        //             (sizeof(ValueT) > 4) ? 64 : 128,
        //             (sizeof(ValueT) > 4) ? 6 : 7,
        //             LOAD_LDG,
        //             LOAD_DEFAULT,
        //             (sizeof(ValueT) > 4) ? LOAD_LDG : LOAD_DEFAULT,
        //             (sizeof(ValueT) > 4) ? LOAD_LDG : LOAD_DEFAULT,
        //             LOAD_LDG,
        //             (sizeof(ValueT) > 4) ? true : false,
        //             (sizeof(ValueT) > 4) ? BLOCK_SCAN_WARP_SCANS : BLOCK_SCAN_RAKING_MEMOIZE>
        //         SpmvPolicyT;


        //     typedef AgentSegmentFixupPolicy<
        //             128,
        //             3,
        //             BLOCK_LOAD_VECTORIZE,
        //             LOAD_LDG,
        //             BLOCK_SCAN_RAKING_MEMOIZE>
        //         SegmentFixupPolicyT;
        // };


        /// SM60
    struct Policy600
    {
        typedef AgentSpmvPolicy<    
                // (sizeof(ValueT) > 4) ? 64 : 128,
                THREADS_PER_BLOCK,
                (sizeof(ValueT) > 4) ? 5 : 10,//7,
                // (sizeof(ValueT) > 4) ? 4 : 7,//7,
                // (sizeof(ValueT) > 4) ? 4 : 7,//7,
                // 7,
                // 10,
                LOAD_DEFAULT,
                LOAD_DEFAULT,
                LOAD_DEFAULT,
                LOAD_DEFAULT,
                LOAD_DEFAULT,
                false,
                BLOCK_SCAN_WARP_SCANS>
            SpmvPolicyT;


        typedef AgentSegmentFixupPolicy<
                THREADS_PER_BLOCK,
                // 64,
                3,
                // 6,
                BLOCK_LOAD_DIRECT,
                LOAD_LDG,
                BLOCK_SCAN_WARP_SCANS>
            SegmentFixupPolicyT;
    };

        /// SM60
        // struct Policy800
        // {
        //     typedef AgentSpmvPolicy<    
        //             // (sizeof(ValueT) > 4) ? 64 : 128,
        //             // THREADS_PER_BLOCK,
        //             THREADS_PER_BLOCK,//7,
        //             // (sizeof(ValueT) > 4) ? 5 : 10,//7,
        //             (sizeof(ValueT) > 4) ? 5 : 10,//7,
        //             // (sizeof(ValueT) > 4) ? 4 : 7,//7,
        //             // 7,
        //             // 10,
        //             LOAD_DEFAULT,
        //             LOAD_DEFAULT,
        //             LOAD_DEFAULT,
        //             LOAD_DEFAULT,
        //             LOAD_DEFAULT,
        //             false,
        //             BLOCK_SCAN_WARP_SCANS>
        //         SpmvPolicyT;


        //     typedef AgentSegmentFixupPolicy<
        //             THREADS_PER_BLOCK,
        //             // 64,
        //             3,
        //             // 6,
        //             BLOCK_LOAD_DIRECT,
        //             LOAD_LDG,
        //             BLOCK_SCAN_WARP_SCANS>
        //         SegmentFixupPolicyT;
        // };


        //---------------------------------------------------------------------
        // Tuning policies of current PTX compiler pass
        //---------------------------------------------------------------------

    // #if (CUB_PTX_ARCH >= 800)
    //     typedef Policy800 PtxPolicy;
    // #else
        typedef Policy600 PtxPolicy;
    // #endif
    // #elif (CUB_PTX_ARCH >= 500)
    //     typedef Policy500 PtxPolicy;

    // #elif (CUB_PTX_ARCH >= 370)
    //     typedef Policy370 PtxPolicy;

    // #else
    //     typedef Policy350 PtxPolicy;

    // #endif

    // "Opaque" policies (whose parameterizations aren't reflected in the type signature)
    struct PtxSpmvPolicyT : PtxPolicy::SpmvPolicyT {};
    struct PtxSegmentFixupPolicy : PtxPolicy::SegmentFixupPolicyT {};


    //---------------------------------------------------------------------
    // Utilities
    //---------------------------------------------------------------------

    /**
        * Initialize kernel dispatch configurations with the policies corresponding to the PTX assembly we will use
        */
    template <typename KernelConfig>
    CUB_RUNTIME_FUNCTION __forceinline__
    static void InitConfigs(
        int             ptx_version,
        KernelConfig    &spmv_config,
        KernelConfig    &segment_fixup_config)
    {
        if (CUB_IS_DEVICE_CODE)
        {
            #if CUB_INCLUDE_DEVICE_CODE
                // We're on the device, so initialize the kernel dispatch configurations with the current PTX policy
                spmv_config.template Init<PtxSpmvPolicyT>();
                segment_fixup_config.template Init<PtxSegmentFixupPolicy>();
            #endif
        }
        else
        {
            #if CUB_INCLUDE_HOST_CODE
                // We're on the host, so lookup and initialize the kernel dispatch configurations with the policies that match the device's PTX version
                // if (ptx_version >= 600)
                {
                    spmv_config.template            Init<typename Policy600::SpmvPolicyT>();
                    segment_fixup_config.template   Init<typename Policy600::SegmentFixupPolicyT>();
                }
                // else if (ptx_version >= 500)
                // {
                //     spmv_config.template            Init<typename Policy500::SpmvPolicyT>();
                //     segment_fixup_config.template   Init<typename Policy500::SegmentFixupPolicyT>();
                // }
                // else if (ptx_version >= 370)
                // {
                //     spmv_config.template            Init<typename Policy370::SpmvPolicyT>();
                //     segment_fixup_config.template   Init<typename Policy370::SegmentFixupPolicyT>();
                // }
                // else
                // {
                //     spmv_config.template            Init<typename Policy350::SpmvPolicyT>();
                //     segment_fixup_config.template   Init<typename Policy350::SegmentFixupPolicyT>();
                // }
            #endif
        }
    }


    /**
        * Kernel kernel dispatch configuration.
        */
    struct KernelConfig
    {
        int block_threads;
        int items_per_thread;
        int tile_items;

        template <typename PolicyT>
        CUB_RUNTIME_FUNCTION __forceinline__
        void Init()
        {
            block_threads       = PolicyT::BLOCK_THREADS;
            items_per_thread    = PolicyT::ITEMS_PER_THREAD;
            tile_items          = block_threads * items_per_thread;
        }
    };


    //---------------------------------------------------------------------
    // Dispatch entrypoints
    //---------------------------------------------------------------------


    //---------------------------------------------------------------------
    // Dispatch entrypoints
    //---------------------------------------------------------------------

    /**
     * Internal dispatch routine for computing a device-wide reduction using the
     * specified kernel functions.
     *
     * If the input is larger than a single tile, this method uses two-passes of
     * kernel invocations.
     */
    template <
        typename                Spmv1ColKernelT,                    ///< Function type of cub::DeviceSpmv1ColKernel
        typename                SpmvSearchKernelT,                  ///< Function type of cub::AgentSpmvSearchKernel
        typename                SpmvKernelT,                        ///< Function type of cub::AgentSpmvKernel
        typename                SegmentFixupKernelT>                 ///< Function type of cub::DeviceSegmentFixupKernelT
    CUB_RUNTIME_FUNCTION __forceinline__
    static cudaError_t InitDispatch(
        void*                   d_temp_storage,                     ///< [in] %Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t&                 temp_storage_bytes,                 ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        SpmvParamsT&            spmv_params,                        ///< SpMV input parameter bundle
        cudaStream_t            stream,                             ///< [in] CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                    debug_synchronous,                  ///< [in] Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.
        Spmv1ColKernelT         spmv_1col_kernel,                   ///< [in] Kernel function pointer to parameterization of DeviceSpmv1ColKernel
        SpmvSearchKernelT       spmv_search_kernel,                 ///< [in] Kernel function pointer to parameterization of AgentSpmvSearchKernel
        SpmvKernelT             spmv_kernel,                        ///< [in] Kernel function pointer to parameterization of AgentSpmvKernel
        SegmentFixupKernelT     segment_fixup_kernel,               ///< [in] Kernel function pointer to parameterization of cub::DeviceSegmentFixupKernel
        KernelConfig            spmv_config,                        ///< [in] Dispatch parameters that match the policy that \p spmv_kernel was compiled for
        KernelConfig            segment_fixup_config)               ///< [in] Dispatch parameters that match the policy that \p segment_fixup_kernel was compiled for
    {
        #ifndef CUB_RUNTIME_ENABLED

        // Kernel launch not supported from this device
            return CubDebug(cudaErrorNotSupported );

        #else
        cudaError error = cudaSuccess;
        do
        {
            // int count=0;
            if (spmv_params.num_cols == 1)
            {

                if (d_temp_storage == NULL)
                {
                    // Return if the caller is simply requesting the size of the storage allocation
                    temp_storage_bytes = 1;
                    break;
                }

                // Get search/init grid dims
                int degen_col_kernel_block_size = INIT_KERNEL_THREADS;
                int degen_col_kernel_grid_size = cub::DivideAndRoundUp(spmv_params.num_rows, degen_col_kernel_block_size);

                if (debug_synchronous) _CubLog("Invoking spmv_1col_kernel<<<%d, %d, 0, %lld>>>()\n",
                    degen_col_kernel_grid_size, degen_col_kernel_block_size, (long long) stream);

                // Invoke spmv_search_kernel
                thrust::cuda_cub::launcher::triple_chevron(
                    degen_col_kernel_grid_size, degen_col_kernel_block_size, 0,
                    stream
                ).doit(spmv_1col_kernel,
                    spmv_params);

                // Check for failure to launch
                if (CubDebug(error = cudaPeekAtLastError())) break;

                // Sync the stream if specified to flush runtime errors
                if (debug_synchronous && (CubDebug(error = SyncStream(stream)))) break;

                break;
            }

            // Get device ordinal
            int device_ordinal;
            if (CubDebug(error = cudaGetDevice(&device_ordinal))) break;

            // Get SM count
            int sm_count;
            if (CubDebug(error = cudaDeviceGetAttribute (&sm_count, cudaDevAttrMultiProcessorCount, device_ordinal))) break;

            // Get max x-dimension of grid
            int max_dim_x;
            if (CubDebug(error = cudaDeviceGetAttribute(&max_dim_x, cudaDevAttrMaxGridDimX, device_ordinal))) break;;

            // Total number of spmv work items
            int num_merge_items = spmv_params.num_rows + spmv_params.num_nonzeros;

            // Tile sizes of kernels
            int blk_merge_tile_size              = spmv_config.block_threads * spmv_config.items_per_thread;
            int segment_fixup_tile_size     = segment_fixup_config.block_threads * segment_fixup_config.items_per_thread;


            // Number of tiles for kernels
            int blk_num_merge_tiles            = cub::DivideAndRoundUp(num_merge_items, blk_merge_tile_size);
            int num_segment_fixup_tiles    = cub::DivideAndRoundUp(blk_num_merge_tiles, segment_fixup_tile_size);

            // Get SM occupancy for kernels
            int spmv_sm_occupancy;
            if (CubDebug(error = MaxSmOccupancy(
                spmv_sm_occupancy,
                spmv_kernel,
                spmv_config.block_threads))) break;

            int segment_fixup_sm_occupancy;
            if (CubDebug(error = MaxSmOccupancy(
                segment_fixup_sm_occupancy,
                segment_fixup_kernel,
                segment_fixup_config.block_threads))) break;


            cudaDeviceProp deviceProp;
            checkCudaErrors(cudaGetDeviceProperties(&deviceProp, 0));
            int numSms = deviceProp.multiProcessorCount;

            int numBlocksPerSm = 0;
            checkCudaErrors(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                            &numBlocksPerSm, spmv_kernel, spmv_config.block_threads, 0));
            dim3 dimGrid(numSms * numBlocksPerSm, 1, 1),
            dimBlock(spmv_config.block_threads, 1, 1);
            
            checkCudaErrors(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                            &numBlocksPerSm, segment_fixup_kernel, segment_fixup_config.block_threads, 0));
            dim3 dimGrid2(numSms * numBlocksPerSm, 1, 1),
            dimBlock2(segment_fixup_config.block_threads, 1, 1);

            // Get grid dimensions
            dim3 spmv_grid_size(
                CUB_MIN(blk_num_merge_tiles, max_dim_x),
                cub::DivideAndRoundUp(blk_num_merge_tiles, max_dim_x),
                1);

            dim3 segment_fixup_grid_size(
                CUB_MIN(num_segment_fixup_tiles, max_dim_x),
                cub::DivideAndRoundUp(num_segment_fixup_tiles, max_dim_x),
                1);

            // Get the temporary storage allocation requirements
            size_t allocation_sizes[3];
            if (CubDebug(error = ScanTileStateT::AllocationSize(num_segment_fixup_tiles, allocation_sizes[0]))) break;    // bytes needed for reduce-by-key tile status descriptors
            allocation_sizes[1] = blk_num_merge_tiles * sizeof(KeyValuePairT);       // bytes needed for block carry-out pairs
            allocation_sizes[2] = (blk_num_merge_tiles + 1) * sizeof(CoordinateT);   // bytes needed for tile starting coordinates
            // fprintf(stderr,"%d\n",count++);

            // Alias the temporary allocations from the single storage blob (or compute the necessary size of the blob)
            void* allocations[3] = {};
            if (CubDebug(error = AliasTemporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes))) break;
            if (d_temp_storage == NULL)
            {
                // Return if the caller is simply requesting the size of the storage allocation
                break;
            }
      
        }
        while (0);

        return error;

        #endif // CUB_RUNTIME_ENABLED
    }


    /**
     * Internal dispatch routine for computing a device-wide reduction
     */
    CUB_RUNTIME_FUNCTION __forceinline__
    static cudaError_t InitDispatch(
        void*                   d_temp_storage,                     ///< [in] %Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t&                 temp_storage_bytes,                 ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        SpmvParamsT&            spmv_params,                        ///< SpMV input parameter bundle
        cudaStream_t            stream                  = 0,        ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                    debug_synchronous       = false)    ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  May cause significant slowdown.  Default is \p false.
    {
        cudaError error = cudaSuccess;
        do
        {
            // Get PTX version
            // printf("!!!!");
            int ptx_version = 0;
            if (CubDebug(error = PtxVersion(ptx_version))) break;
            // printf("????");

            // Get kernel kernel dispatch configurations
            KernelConfig spmv_config, segment_fixup_config;
            InitConfigs(ptx_version, spmv_config, segment_fixup_config);

            if (CubDebug(error = InitDispatch(
                d_temp_storage, temp_storage_bytes, spmv_params, stream, debug_synchronous,
                DeviceSpmv1ColKernel<PtxSpmvPolicyT, ValueT, OffsetT>,
                DeviceSpmvSearchKernel<PtxSpmvPolicyT, OffsetT, CoordinateT, SpmvParamsT>,
                // DeviceSpmvKernel<PtxSpmvPolicyT, ScanTileStateT, ValueT, OffsetT, CoordinateT, false, false>,
                MyDeviceSpmvKernel<PtxSpmvPolicyT, ScanTileStateT, ValueT, OffsetT, CoordinateT, false, false>,
                // DeviceSegmentFixupKernel<PtxSegmentFixupPolicy, KeyValuePairT*, ValueT*, OffsetT, ScanTileStateT>,
                MyDeviceSegmentFixupKernel<PtxSegmentFixupPolicy, KeyValuePairT*, ValueT*, OffsetT, ScanTileStateT>,
                spmv_config, segment_fixup_config))) break;

        }
        while (0);

        return error;
    }

    CUB_RUNTIME_FUNCTION __forceinline__
    static cudaError_t ProcessDispatchCG(
            // void * kernelArgs[14],
            // size_t sMemSize
            SMCacheParams<OffsetT,size_t> &smParamsT,
            CgParams<ValueT,double,OffsetT> &cgParamsT,
            // SMCacheParams<OffsetT,size_t> smParamsT,
            //initialized temp storyage
            void *d_temp_storage,
            size_t temp_storage_bytes,
            SpmvParams<ValueT,OffsetT> spmvParams,
            bool isStaticIter=false
            )
        {

            cudaError error = cudaSuccess;
            {
                cudaDeviceProp deviceProp;
                checkCudaErrors(cudaGetDeviceProperties(&deviceProp, 0));

                int ptx_version = 0;

                error = PtxVersion(ptx_version);

                KernelConfig spmv_config, segment_fixup_config;
                InitConfigs(ptx_version, spmv_config, segment_fixup_config);

                
                auto spmv_search_kernel=DeviceSpmvSearchKernel<PtxSpmvPolicyT, OffsetT, CoordinateT, SpmvParamsT>;
                auto spmv_kernel=DeviceSpmvKernel<PtxSpmvPolicyT, ScanTileStateT, ValueT, OffsetT, CoordinateT, false, false>;
                auto segment_fixup_kernel=DeviceSegmentFixupKernel<PtxSegmentFixupPolicy, KeyValuePairT*, ValueT*, OffsetT, ScanTileStateT>;
//baseline

                auto perk_cg_kernel = isStaticIter? gpuConjugateGradient_cub
                        <PtxSpmvPolicyT, ScanTileStateT, ValueT, OffsetT, CoordinateT, 
                        false, false, PtxSegmentFixupPolicy, ValueT*, 
                        isbaseline,cacheMatrix,cacheVector,true>:gpuConjugateGradient_cub
                        <PtxSpmvPolicyT, ScanTileStateT, ValueT, OffsetT, CoordinateT, 
                        false, false, PtxSegmentFixupPolicy, ValueT*, 
                        isbaseline,cacheMatrix,cacheVector,false>;  

                using std::is_same;
                size_t unchangeableTempt=0;//max(sizeof(AgentSpmvT::TempStorage),sizeof(AgentSegmentFixupT::TempStorage));


                OffsetT tile_size;
                OffsetT MatrixUnitSize;
                if(is_same<ValueT, float>::value) { 
                  // printf("is float\n");
                  typedef AgentSpmv<
                        DispatchCG<float,int,true,false,false>::PtxPolicy::SpmvPolicyT,
                        // PtxSpmvPolicyT,
                        // PtxSpmvPolicyT,
                        float,
                        int,
                        true,
                        false>
                    AgentSpmvT;
                    // typedef 
                    typedef AgentSegmentFixup<
                                    DispatchCG<float,int,true,false,false>::PtxPolicy::SegmentFixupPolicyT,
                                    // PtxSegmentFixupPolicy,
                                    // PtxSegmentFixupPolicy,
                                    KeyValuePair<int,float>*,
                                    float*,
                                    cub::Equality,
                                    cub::Sum,
                                    int>
                    AgentSegmentFixupT;
                    unchangeableTempt=max(sizeof(AgentSpmvT::TempStorage),sizeof(AgentSegmentFixupT::TempStorage));
                    // unchangeableTempt=(sizeof(AgentSpmvT::TempStorage))+2*(sizeof(AgentSegmentFixupT::TempStorage));

                    /* stuff */
                    tile_size=AgentSpmvT::ITEMS_PER_THREAD*THREADS_PER_BLOCK;
                    MatrixUnitSize=sizeof(typename AgentSpmvT::MatrixTileUnit);
                }
                else
                {
                   // printf("is double\n");
                   typedef AgentSpmv<
                        DispatchCG<double,int,true,false,false>::PtxPolicy::SpmvPolicyT,
                        // PtxSpmvPolicyT,
                        // PtxSpmvPolicyT,
                        double,
                        int,
                        true,
                        false>
                    AgentSpmvT;
                    // typedef 
                    typedef AgentSegmentFixup<
                                    DispatchCG<double,int,true,false,false>::PtxPolicy::SegmentFixupPolicyT,
                                    // PtxSegmentFixupPolicy,
                                    // PtxSegmentFixupPolicy,
                                    KeyValuePair<int,double>*,
                                    double*,
                                    cub::Equality,
                                    cub::Sum,
                                    int>
                    AgentSegmentFixupT;
                    unchangeableTempt=max(sizeof(AgentSpmvT::TempStorage),sizeof(AgentSegmentFixupT::TempStorage));
                    // unchangeableTempt=(sizeof(AgentSpmvT::TempStorage))+(sizeof(AgentSegmentFixupT::TempStorage));
                    tile_size=AgentSpmvT::ITEMS_PER_THREAD*THREADS_PER_BLOCK;
                    MatrixUnitSize=sizeof(typename AgentSpmvT::MatrixTileUnit);
                }

                

                size_t sMemSize = sizeof(double) * ((THREADS_PER_BLOCK) + 1  +1);
                size_t launcableTmep=sMemSize;
                int numBlocksPerSm = 0;
                int numThreads = THREADS_PER_BLOCK;
                size_t sMemSizeTempt=max(launcableTmep,unchangeableTempt);
                
                #ifndef __PRINT__
                    printf("tile size is %d\n",tile_size);
                    printf("<%ld,%ld>\n",launcableTmep,unchangeableTempt);
                    printf("launchable tmp memr is %lu\n",launcableTmep);
                    printf("tmp memr is %lu\n",sMemSizeTempt);
                #endif
                int num_merge_items = spmvParams.num_rows + spmvParams.num_nonzeros;
                int blk_merge_tile_size = spmv_config.block_threads * spmv_config.items_per_thread;
                int blk_num_merge_tiles = cub::DivideAndRoundUp(num_merge_items, blk_merge_tile_size);

                checkCudaErrors(cudaGetDeviceProperties(&deviceProp, 0));
                int numSms = deviceProp.multiProcessorCount;
                
                int maxSharedMemory;
                cudaDeviceGetAttribute (&maxSharedMemory, cudaDevAttrMaxSharedMemoryPerMultiprocessor,0 );

                int assumBlkPerSm=5;

                int SharedMemoryUsed = maxSharedMemory;
                OffsetT sm_size_coor = 0;
                OffsetT sm_coor = 0;
                OffsetT smBlkMatrixUnitNum = 0;
                OffsetT sm_num_r = 0;
                OffsetT sm_blk_size_r= 0;
                while(true)
                {
                    cudaFuncSetAttribute(perk_cg_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SharedMemoryUsed);
                    // #ifndef __PRINT__
                        printf("current max shared memory is %d\n", SharedMemoryUsed);
                    // #endif
                    checkCudaErrors(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                        &numBlocksPerSm, perk_cg_kernel, numThreads, sMemSizeTempt+sizeof(CoordinateT)* 2*cub::DivideAndRoundUp(blk_num_merge_tiles,assumBlkPerSm*numSms)));

                    dim3 dimGrid(numSms * numBlocksPerSm, 1, 1);
                    // #ifndef __PRINT__
                    printf("<%dX%d>\n",numBlocksPerSm*numSms,numThreads);
                    if(numBlocksPerSm!=assumBlkPerSm)
                    {
                        assumBlkPerSm=numBlocksPerSm ;
                        printf("0 current blkpsm is %d\n",assumBlkPerSm);
                        continue;
                    }
                    // #endif
                    
                    OffsetT sMemSizeUsable=SharedMemoryUsed/numBlocksPerSm;//- unchangeableTempt;//(sizeof(AgentSpmvT::TempStorage)+sizeof(AgentSegmentFixupT::TempStorage))*3;;//unchangeableTempt;

                    sm_coor=isbaseline?0: 2*cub::DivideAndRoundUp(blk_num_merge_tiles,dimGrid.x);
                 
                    sm_size_coor = sm_coor*sizeof(CoordinateT);

                    OffsetT sm_num_r_remained=MAX(sMemSizeUsable-sMemSizeTempt-sm_size_coor,0)/sizeof(ValueT)/(THREADS_PER_BLOCK);
                    OffsetT sm_num_r_preffered = cacheVector? cgParamsT.N/numBlocksPerSm/numSms/THREADS_PER_BLOCK:0;
                    sm_num_r = MIN(sm_num_r_preffered,sm_num_r_remained);
                    sm_blk_size_r = sm_num_r*sizeof(ValueT)*spmv_config.block_threads;
                    sMemSize=sMemSizeTempt+sm_size_coor+sm_blk_size_r;

                    OffsetT sMemSizeRmained=sMemSizeUsable-sMemSize;//sMemSizeTempt-sm_size_coor;
                    OffsetT smBlkMatrixNumPreffered=cacheMatrix? cub::DivideAndRoundUp(blk_num_merge_tiles,dimGrid.x):0;
                    smBlkMatrixUnitNum=sMemSizeRmained/MatrixUnitSize;
                    smBlkMatrixUnitNum=min(smBlkMatrixNumPreffered,smBlkMatrixUnitNum);
                    OffsetT smBlkMatrixTotalSize = smBlkMatrixUnitNum*MatrixUnitSize;

                    sMemSize=sMemSizeTempt+sm_size_coor+smBlkMatrixTotalSize+sm_blk_size_r;
                    //check if can break;
                    checkCudaErrors(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                        &numBlocksPerSm, perk_cg_kernel, numThreads, sMemSize));
                    if(numBlocksPerSm!=assumBlkPerSm)
                    {
                        assumBlkPerSm=numBlocksPerSm;
                        printf("1 current blkpsm is %d\n",assumBlkPerSm);
                        SharedMemoryUsed-=1024;
                    }
                    else
                    {
                        break;
                    }
                }
                dim3 dimGrid(numSms * assumBlkPerSm, 1, 1);
                #ifndef __PRINT__
                    printf("%d,%d,%ld()()()\n",sm_blk_size_r, sm_num_r, sMemSize);
                    printf("[%d,%d:%d\\%d]\n",smBlkMatrixTotalSize, sm_blk_size_r,(int)sMemSize,sMemSizeUsable);

                    printf("%d\n",num_merge_items);
                    printf("%d\n",spmv_config.items_per_thread);
                    printf("%f\n",(double)sizeof(OffsetT)*spmvParams.num_rows/1024/1024);
                    printf("unchangeable %f KB",(double)unchangeableTempt/1024);
                    printf("<%d,%d>\n",smBlkMatrixNumPreffered,smBlkMatrixUnitNum);
                    printf("SIZE<%f,%f MB,%f MB, %f MB>\n",
                                    (double)MatrixUnitSize,
                                    (double)smBlkMatrixUnitNum*MatrixUnitSize*numSms*numBlocksPerSm/1024/1024,
                                    (double)spmvParams.num_nonzeros*(sizeof(OffsetT)+sizeof(ValueT))/1024/1024,
                                    // (double)(cub::DivideAndRoundUp(spmvParams.num_nonzeros, blk_merge_tile_size)-1)*(256*8*10)/1024/1024
                                    (double)(smBlkMatrixNumPreffered-1)*numSms*numBlocksPerSm*MatrixUnitSize/1024/1024
                                );
                    printf("<%d,%d>\n",smBlkMatrixNumPreffered,smBlkMatrixUnitNum);
                #endif
          
                int blockCountSMX = cub::DivideAndRoundUp(blk_num_merge_tiles,dimGrid.x);

                smParamsT.sm_size_coor=sm_size_coor;
                smParamsT.sm_size_total_coor=sm_size_coor*dimGrid.x;
                smParamsT.sm_size_unit_coor=sm_coor;
                
                smParamsT.sm_size_unit_matrix=MatrixUnitSize;
                smParamsT.sm_size_blocktile_matrix=MatrixUnitSize* spmv_config.block_threads;
                smParamsT.sm_num_matrixperblk=smBlkMatrixUnitNum;
                smParamsT.sm_size_blocktile_total_matrix=smBlkMatrixUnitNum*smParamsT.sm_size_blocktile_matrix;

                smParamsT.sm_num_r=sm_num_r;
                smParamsT.sm_blk_size_r=sm_blk_size_r;
 
                smParamsT.sMemSize=sMemSize;
                smParamsT.sMemSizeTempt=sMemSizeTempt;
                smParamsT.sMemSizeTemptTotal=sMemSizeTempt;
     
                cudaStream_t stream=0;
                //bool debug_synchronous=false;
                {
                    do
                    {
                        assert(spmvParams.num_cols != 1);
                        // Get device ordinal
                        int device_ordinal;
                        if (CubDebug(error = cudaGetDevice(&device_ordinal))) break;

                        // Get SM count
                        int sm_count;
                        if (CubDebug(error = cudaDeviceGetAttribute (&sm_count, cudaDevAttrMultiProcessorCount, device_ordinal))) break;

                        // Get max x-dimension of grid
                        int max_dim_x;
                        if (CubDebug(error = cudaDeviceGetAttribute(&max_dim_x, cudaDevAttrMaxGridDimX, device_ordinal))) break;;

                        // Total number of spmv work items
                        int num_merge_items = spmvParams.num_rows + spmvParams.num_nonzeros;

                        // Tile sizes of kernels
                        int blk_merge_tile_size             = spmv_config.block_threads * spmv_config.items_per_thread;
                        int segment_fixup_tile_size     = segment_fixup_config.block_threads * segment_fixup_config.items_per_thread;

                        // Number of tiles for kernels
                        int blk_num_merge_tiles            = cub::DivideAndRoundUp(num_merge_items, blk_merge_tile_size);
                        int num_segment_fixup_tiles    = cub::DivideAndRoundUp(blk_num_merge_tiles, segment_fixup_tile_size);

                        int spmv_sm_occupancy;
                        if (CubDebug(error = MaxSmOccupancy(
                            spmv_sm_occupancy,
                            spmv_kernel,
                            spmv_config.block_threads))) break;

                        int segment_fixup_sm_occupancy;
                        if (CubDebug(error = MaxSmOccupancy(
                            segment_fixup_sm_occupancy,
                            segment_fixup_kernel,
                            segment_fixup_config.block_threads))) break;

                        // Get the temporary storage allocation requirements
                        size_t allocation_sizes[3];
                        if (CubDebug(error = ScanTileStateT::AllocationSize(num_segment_fixup_tiles, allocation_sizes[0]))) break;    // bytes needed for reduce-by-key tile status descriptors
                        allocation_sizes[1] = blk_num_merge_tiles * sizeof(KeyValuePairT);       // bytes needed for block carry-out pairs
                        allocation_sizes[2] = (blk_num_merge_tiles + 1) * sizeof(CoordinateT);   // bytes needed for tile starting coordinates

                        // printf("<%lu,%lu,%lu>\n",allocation_sizes[0],allocation_sizes[1],allocation_sizes[2]);

                        // Alias the temporary allocations from the single storage blob (or compute the necessary size of the blob)
                        void* allocations[3] = {};
                        if (CubDebug(error = AliasTemporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes))) break;
                        if (d_temp_storage == NULL)
                        {
                            // Return if the caller is simply requesting the size of the storage allocation
                            break;
                        }

                        // Construct the tile status interface
                        ScanTileStateT tile_state;
                        if (CubDebug(error = tile_state.Init(num_segment_fixup_tiles, allocations[0], allocation_sizes[0]))) break;

                        // Alias the other allocations
                        KeyValuePairT*  d_tile_carry_pairs      = (KeyValuePairT*) allocations[1];  // Agent carry-out pairs
                        CoordinateT*    d_tile_coordinates      = (CoordinateT*) allocations[2];    // Agent starting coordinates

                        // Get search/init grid dims
                        int search_block_size   = 32;
                        int search_grid_size    = cub::DivideAndRoundUp(blk_num_merge_tiles + 1, search_block_size);


                        {
                            spmv_search_kernel<<<search_grid_size, search_block_size, 0, stream>>>
                                (blk_num_merge_tiles,
                                    d_tile_coordinates,
                                    spmvParams);
                        }

                        void *kernelArgs[] = {
                                (void *)&spmvParams,
                                (void *)&d_tile_coordinates,
                                (void *)&d_tile_carry_pairs,
                                (void *)&blk_num_merge_tiles,
                                (void *)&tile_state,
                                (void *)&num_segment_fixup_tiles,
                                // (void *)&num_segment_fixup_tiles,
                                // (void *)&spmvParams.d_vector_y,

                                (void *)&cgParamsT.d_row_offsets,  (void *)&cgParamsT.d_column_indices, (void *)&cgParamsT.d_val, 
                                (void *)&cgParamsT.d_vector_x ,(void *)&cgParamsT.d_vector_Ax , 
                                (void *)&cgParamsT.d_vector_p, (void *)&cgParamsT.d_vector_r,   (void *)&cgParamsT.d_vector_dot_results,
                                (void *)&cgParamsT.nz, (void *)&cgParamsT.N,  
                                (void *)&cgParamsT.tol,

                                (void *)&smParamsT,
                                (void *)&cgParamsT.d_iter,
                                (void *)&cgParamsT.max_iter,
                                (void *)&blockCountSMX
                            };
                        cudaDeviceProp deviceProp;
                        checkCudaErrors(cudaGetDeviceProperties(&deviceProp, 0));
                        size_t sMemSize = smParamsT.sMemSize;
                        // cudaError error = cudaSuccess;
                        int numBlocksPerSm = 0;
                        int numThreads = THREADS_PER_BLOCK;
                        int numSms = deviceProp.multiProcessorCount;
                        
                        // cudaFuncSetAttribute(perk_cg_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,  sMemSize);
                        checkCudaErrors(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                            &numBlocksPerSm, perk_cg_kernel, numThreads, sMemSize));
                        // printf("<%dX%d>\n",numBlocksPerSm*numSms,numThreads);
                        dim3 dimGrid(numSms * numBlocksPerSm, 1, 1),
                        dimBlock(THREADS_PER_BLOCK, 1, 1);
                        cgParamsT.gridDim =dimGrid.x;


                        printf("%dX%d\t",numBlocksPerSm,numThreads);
                        
                        error=(cudaLaunchCooperativeKernel(
                            (void *)perk_cg_kernel,
                            dimGrid, dimBlock, kernelArgs,
                            sMemSize, NULL));
                    }
                    while (0);
                }
            }
            

            return error;
        } 
    };

}

// CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)
