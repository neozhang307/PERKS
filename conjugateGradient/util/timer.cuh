/******************************************************************************
 * Copyright (c) 2011-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIAeBILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

// #pragma once

#if defined(_WIN32) || defined(_WIN64)
    #include <windows.h>
    #undef min       // Windows is terrible for polluting macro namespace
    #undef max       // Windows is terrible for polluting macro namespace
    #undef small     // Windows is terrible for polluting macro namespace
#else
    #include <sys/resource.h>
    #include <time.h>
#endif

#include <stdio.h>
#include <string.h>

#include <map>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <iostream>
#include <limits>

#ifdef CUB_MKL
    #include "omp.h"
#endif



#ifdef CUB_MKL

/**
 * CPU timer (use omp wall timer because rusage accumulates time from all threads)
 */
struct CpuTimer
{
    double start;
    double stop;

    void Start();


    void Stop();

    float ElapsedMillis();
};

#else

struct CpuTimer
{
#if defined(_WIN32) || defined(_WIN64)

    LARGE_INTEGER ll_freq;
    LARGE_INTEGER ll_start;
    LARGE_INTEGER ll_stop;

    CpuTimer();

    void Start();

    void Stop();

    float ElapsedMillis();

#else   // _WINXXX

    rusage start;
    rusage stop;

    void Start();

    void Stop();

    float ElapsedMillis();
#endif  // _WINXXX
};



#endif  // CUB_MKL

#ifdef __NVCC__


/**
 * GPU timer
 */
struct GpuTimer
{
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer();
    ~GpuTimer();
    void Start();
    void Stop();
    float ElapsedMillis();
};


#endif // __NVCC__


