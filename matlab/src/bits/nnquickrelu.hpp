// @file nnquickrelu.hpp
// @brief Quick relu block
// @author Samuel Albanie 
// @author Andrea Vedaldi
/*
Copyright (C) 2017 Samuel Albanie and Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef __vl__nnquickrelu__
#define __vl__nnquickrelu__

#include <bits/data.hpp>
#include <stdio.h>

namespace vl {

  vl::ErrorCode
  nnquickrelu_forward(vl::Context& context,
                      vl::Tensor output,
                      vl::Tensor data,
                      float leak) ;

  vl::ErrorCode
  nnquickrelu_backward(vl::Context& context,
                      vl::Tensor derData,
                      vl::Tensor data,
                      vl::Tensor derOutput,
                      float leak) ;
}

#endif /* defined(__vl__nnquickrelu__) */
