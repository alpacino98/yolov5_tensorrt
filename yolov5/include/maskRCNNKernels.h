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

#ifndef TRT_MASKRCNN_UTILS_H
#define TRT_MASKRCNN_UTILS_H

#include "NvInfer.h"
#include "plugin.h"

using namespace nvinfer1;

// RESIZE NEAREST
void resizeNearest(dim3 grid, dim3 block, cudaStream_t stream, int nbatch, float scale, int2 osize, float const* idata,
    int istride, int ibatchstride, float* odata, int ostride, int obatchstride);
// SPECIAL SLICE
void specialSlice(cudaStream_t stream, int batch_size, int boxes_cnt, const void* idata, void* odata);

#endif // TRT_MASKRCNN_UTILS_H
