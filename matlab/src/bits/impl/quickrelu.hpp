// @file quickrelu.hpp
// @brief quickrelu 
// @author Samuel Albanie

/*
Copyright (C) 2017 Samuel Albanie.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef VL_QUICKRELU_H
#define VL_QUICKRELU_H

#include <bits/data.hpp>
#include <cstddef>

namespace vl { namespace impl {

  template<vl::DeviceType dev, typename T>
  struct quickrelu
  {
    static vl::ErrorCode
    forward(Context& context,
            T* output,
            T const* data,
            T leak, 
            size_t outSize) ;

    static vl::ErrorCode
    backward(Context& context,
             T* derData,
             T const* data,
             T const* derOutput,
             T leak,
             size_t outSize) ;
  } ;

} }
#endif /* defined(VL_QUICKRELU_H) */
