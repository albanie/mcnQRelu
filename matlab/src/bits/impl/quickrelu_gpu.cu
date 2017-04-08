// @file quickrelu_gpu.cu
// @brief quickrelu CUDA implementation 
// (this code is based on the implementation provided in caffe)
// @author Samuel Albanie
// @author Andrea Vedaldi

/*
Copyright (C) 2017 Samuel Albanie and Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "quickrelu.hpp"
#include <bits/data.hpp>
#include <assert.h>
#include <float.h>
#include <cstdio>

/* ------------------------------------------------------------ */
/*                                                      kernels */
/* ------------------------------------------------------------ */

template <typename T>
__global__ void reluForwardKernel(const int numThreads,
                                  const T* in, 
                                  T* out,
                                  T leak) 
{
    // Grid stride-loop 
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; 
             index < numThreads ; 
             index += blockDim.x * gridDim.x) 
    {
        out[index] = in[index] > 0 ? in[index] : in[index] * leak;
    }
}

template <typename T>
__global__ void reluBackwardKernel(const int numThreads,
                                   const T* in, 
                                   const T* der, 
                                   T* out,
                                   T leak) 
{
    // Grid stride-loop 
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; 
             index < numThreads ; 
             index += blockDim.x * gridDim.x) 
    {
        out[index] = in[index] > 0 ? der[index] : der[index] * leak;
    }
}

/* ------------------------------------------------------------ */
/*                                              kernel wrappers */
/* ------------------------------------------------------------ */

template <typename T>
void reluForwardGPU(const int numThreads,
                    const T* in, 
                    T* out,
                    T leak) 
{
    int numBlocks = (numThreads + 511) / 512 ;
    reluForwardKernel<T><<<numBlocks, 512>>>(numThreads, in, out, leak) ;
    cudaError_t status = cudaPeekAtLastError() ;
    if (status != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(status));
        exit(-1) ;
    }
}

template <typename T>
void reluBackwardGPU(const int numThreads,
                    const T* in, 
                    const T* der, 
                    T* out,
                    T leak) 
{
    int numBlocks = (numThreads + 511) / 512 ;
    reluBackwardKernel<T><<<numBlocks, 512>>>(numThreads, in, der, out, leak) ;
    cudaError_t status = cudaPeekAtLastError() ;
    if (status != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(status));
        exit(-1) ;
    }
}


namespace vl { namespace impl {

  template<typename T>
  struct quickrelu<vl::VLDT_GPU, T>
  {

    /* ------------------------------------------------------------ */
    /*                                                      forward */
    /* ------------------------------------------------------------ */
    static vl::ErrorCode
    forward(Context& context,
            T* output,
            T const* data,
            T const leak,
            size_t outSize)
    {
      reluForwardGPU<T>(outSize, data, output, leak) ;

      cudaError_t status = cudaPeekAtLastError() ;
      return (status == cudaSuccess) ? vl::VLE_Success : vl::VLE_Cuda ;
    }


    /*------------------------------------------------------------- */
    /*                                                     backward */
    /* ------------------------------------------------------------ */

    static vl::ErrorCode
    backward(Context& context,
             T* derData,
             T const* data,
             T const* derOutput,
             T const leak,
             size_t outSize)
    {
      reluBackwardGPU<T>(outSize, data, derOutput, derData, leak) ;

      cudaError_t status = cudaPeekAtLastError() ;
      return (status == cudaSuccess) ? vl::VLE_Success : vl::VLE_Cuda ;
    }
  } ;

} } // namespace vl::impl

template struct vl::impl::quickrelu<vl::VLDT_GPU, float> ;

#ifdef ENABLE_DOUBLE
template struct vl::impl::quickrelu<vl::VLDT_GPU, double> ;
#endif
