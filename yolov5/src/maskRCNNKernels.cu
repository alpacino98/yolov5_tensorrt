/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <assert.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "maskRCNNKernels.h"
#include "plugin.h"
#include <NvInfer.h>
#include <assert.h>
#include <iostream>
#include <stdio.h>
// #include <thrust/device_ptr.h>
// #include <thrust/fill.h>

#define DUBUG_KERNEL 0
#define DUBUG_BATCH 0
#define DEBUG_T 1

#define dMIN(a, b) ((a) < (b) ? (a) : (b))
#define dMAX(a, b) ((a) > (b) ? (a) : (b))
#define dCLAMP(x, xMin, xMax) ((x) > (xMin) ? ((x) < (xMax) ? (x) : (xMax)) : (xMin))



__global__ void resize_nearest_kernel_2d(int nbatch, float scale, int2 osize, float const* idata, int istride,
    int ibatchstride, float* odata, int ostride, int obatchstride)
{

    int x0 = threadIdx.x + blockIdx.x * blockDim.x;
    int y0 = threadIdx.y + blockIdx.y * blockDim.y;
    int z0 = blockIdx.z;
    for (int batch = z0; batch < nbatch; batch += gridDim.z)
    {
        for (int oy = y0; oy < osize.y; oy += blockDim.y * gridDim.y)
        {
            for (int ox = x0; ox < osize.x; ox += blockDim.x * gridDim.x)
            {
                int ix = int(ox / scale);
                int iy = int(oy / scale);
                odata[batch * obatchstride + oy * ostride + ox] = idata[batch * ibatchstride + iy * istride + ix];
            }
        }
    }
}

void resizeNearest(dim3 grid, dim3 block, cudaStream_t stream, int nbatch, float scale, int2 osize, float const* idata,
    int istride, int ibatchstride, float* odata, int ostride, int obatchstride)
{

    resize_nearest_kernel_2d<<<grid, block, 0, stream>>>(
        nbatch, scale, osize, idata, istride, ibatchstride, odata, ostride, obatchstride);
}


