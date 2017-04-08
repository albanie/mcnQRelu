// @file nnquickrelu.cu
// @brief Multibox Detector block
// @author Samuel Albanie
// @author Andrea Vedaldi

/*
Copyright (C) 2017- Samuel Albanie and Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "nnquickrelu.hpp"
#include "impl/quickrelu.hpp"

#if ENABLE_GPU
#include <bits/datacu.hpp>
#endif

#include <cstdio>
#include <assert.h>

using namespace vl ;

/* ---------------------------------------------------------------- */
/*                                                quickrelu_forward */
/* ---------------------------------------------------------------- */

#define DISPATCH(deviceType, T) \
error = vl::impl::quickrelu<deviceType, T>::forward \
(context, \
(T*) output.getMemory(), \
(T const*) data.getMemory(), \
(T) leak, \
output.getHeight(), \
output.getWidth(), \
output.getDepth(), \
output.getSize()) ;

#define DISPATCH2(deviceType) \
switch (T) { \
  case VLDT_Float : DISPATCH(deviceType, float) ; \
    break ; \
  IF_DOUBLE(case VLDT_Double : DISPATCH(deviceType, double) ; \
    break ;) \
  default: \
    assert(false) ; \
  return VLE_Unknown ; \
}

vl::ErrorCode
vl::nnquickrelu_forward(Context& context,
                   vl::Tensor output, 
                   vl::Tensor data, 
                   float leak)
{
  vl::ErrorCode error = VLE_Success ;
  vl::DataType T = output.getDataType() ;

  switch (output.getDeviceType()) {
    default:
      assert(false) ;
      error = vl::VLE_Unknown ;
      break ;

    case vl::VLDT_CPU:
      DISPATCH2(vl::VLDT_CPU) ;
      break ;

#if ENABLE_GPU
    case vl::VLDT_GPU:
      DISPATCH2(vl::VLDT_GPU) ;
      break ;
#endif
  }
  return error ;
}

/* ---------------------------------------------------------------- */
/*                                             nnquickrelu_backward */
/* ---------------------------------------------------------------- */

#undef DISPATCH
#undef DISPATCH2

#define DISPATCH(deviceType, T) \
error = vl::impl::quickrelu<deviceType, T>::backward \
(context, \
(T*) derData.getMemory(), \
(T const*) data.getMemory(), \
(T const*) derOutput.getMemory(), \
leak, \
derOutput.getHeight(), \
derOutput.getWidth(), \
derOutput.getDepth(), \
derOutput.getSize()) ;

#define DISPATCH2(deviceType) \
switch (T) { \
  case VLDT_Float : DISPATCH(deviceType, float) ; \
    break ; \
  IF_DOUBLE(case VLDT_Double : DISPATCH(deviceType, double) ; break ;) \
  default: \
    assert(false) ; \
  return VLE_Unknown ; \
}

vl::ErrorCode
vl::nnquickrelu_backward(Context& context,
                        vl::Tensor derData,
                        vl::Tensor data, 
                        vl::Tensor derOutput,
                        float leak)
{
  vl::ErrorCode error = vl::VLE_Success ;
  vl::DataType T = derOutput.getDataType() ;

  switch (derOutput.getDeviceType()) {
    default:
      assert(false) ;
      error = vl::VLE_Unknown ;
      break ;

    case vl::VLDT_CPU:
      DISPATCH2(vl::VLDT_CPU) ;
      break ;

#if ENABLE_GPU
    case vl::VLDT_GPU:
      DISPATCH2(vl::VLDT_GPU) ;
      break ;
#endif
  }
  return error ;
}
