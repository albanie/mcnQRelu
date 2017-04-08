// @file quickrelu_gpu.cu
// @brief quickrelu CUDA implementation
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

namespace vl { namespace impl {

  template<typename T>
  struct quickrelu<vl::VLDT_CPU, T>
  {

    /* ------------------------------------------------------------ */
    /*                                                      forward */
    /* ------------------------------------------------------------ */
    static vl::ErrorCode
    forward(Context& context,
            T* output,
            T const* data,
            float const leak,
            size_t outWidth,
            size_t outHeight,
            size_t outDepth,
            size_t batchSize)
    {
      size_t size = outWidth * outHeight * outDepth * batchSize ;
      T zero = T(0) ;
      if (abs(leak) > 1e7) {
        for (int i = 0 ; i < size ; ++i) {
          T in = data[i] ;
          output[i] = std::max(in, zero) + leak * std::min(zero, in) ;
        }
      } else {
        for (int i = 0 ; i < size ; ++i) {
          output[i] = std::max(data[i], zero) ;
        }
      }
      return VLE_Success ;
    }


    /*------------------------------------------------------------- */
    /*                                                     backward */
    /* ------------------------------------------------------------ */

    static vl::ErrorCode
    backward(Context& context,
             T* derData,
             T const* data,
             T const* derOutput,
             float const leak,
             size_t outWidth,
             size_t outHeight,
             size_t outDepth,
             size_t batchSize)
    {
      size_t size = outWidth * outHeight * outDepth * batchSize ;
      T zero = T(0) ;
      if (abs(leak) > 1e7) {
        for (int i = 0 ; i < size ; ++i) {
          T in = data[i] ;
          derData[i] = derOutput[i] * (in > 0) + leak * (in <= 0) ;
        }
      } else {
        for (int i = 0 ; i < size ; ++i) {
          derData[i] = derOutput[i] * (data[i] > 0) ;
        }
      }
      return VLE_Success ;
    }
  } ;

} } // namespace vl::impl

template struct vl::impl::quickrelu<vl::VLDT_CPU, float> ;

#ifdef ENABLE_DOUBLE
template struct vl::impl::quickrelu<vl::VLDT_CPU, double> ;
#endif
