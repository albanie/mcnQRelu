// @file vl_nnquickrelu.cu
// @brief quick relu block MEX wrapper
// @author Samuel Albanie 
/*
Copyright (C) 2017 Samuel Albanie and Andrea Vedaldi
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include <bits/mexutils.h>
#include <bits/datamex.hpp>
#include "bits/nnquickrelu.hpp"

#if ENABLE_GPU
#include <bits/datacu.hpp>
#endif

#include <assert.h>

/* option codes */
enum {
  opt_leak = 0,
  opt_verbose,
  opt_no_der_data,
} ;

/* options */
VLMXOption  options [] = {
  {"Leak",            1,   opt_leak       },
  {"Verbose",         0,   opt_verbose    },
  {"NoDerData",       0,   opt_no_der_data},
  {0,                 0,   0              }
} ;

/* ---------------------------------------------------------------- */
/*                                                          Context */
/* ---------------------------------------------------------------- */

vl::MexContext context ;

/*
 Resetting the context here resolves a crash when MATLAB quits and
 the ~Context function is implicitly called on unloading the MEX file.
 */
void atExit()
{
  context.clear() ;
}

/* ---------------------------------------------------------------- */
/*                                                       MEX driver */
/* ---------------------------------------------------------------- */

enum {
  IN_DATA = 0, IN_DEROUTPUT, IN_END
} ;

enum {
  OUT_RESULT = 0, OUT_END
} ;

void mexFunction(int nout, mxArray *out[],
                 int nin, mxArray const *in[])
{
  float leak = 0 ;
  int verbosity = 0 ;
  int opt ;
  bool backMode = false ;
  bool computeDerData = true ;
  int next = IN_END ;
  mxArray const *optarg ;

  /* -------------------------------------------------------------- */
  /*                                            Check the arguments */
  /* -------------------------------------------------------------- */

  if (nin < 1) {
    mexErrMsgTxt("There are not enough arguments.") ;
  }

  if (nin > 1 && vlmxIsString(in[1],-1)) {
    next = 3 ;
    backMode = 0 ;
  } else {
    backMode = (nin >= 2) ;
  }

  while ((opt = vlmxNextOption (in, nin, options, &next, &optarg)) >= 0) {
    switch (opt) {
      case opt_verbose :
        ++ verbosity ;
        break ;

      case opt_leak :
        if (!vlmxIsScalar(optarg)) {
          vlmxError(VLMXE_IllegalArgument, "LEAK is not a scalar.") ;
        }
        leak = (float)mxGetPr(optarg)[0] ;
        break ;

      case opt_no_der_data :
        computeDerData = false ;
        break ;

      default: 
        break ;
    }
  }

  vl::MexTensor data(context) ;
  vl::MexTensor derOutput(context) ;

  data.init(in[IN_DATA]) ;
  data.reshape(4) ;

  /* check for GPU/data class consistency */
  if (!vl::areCompatible(data, derOutput)) {
    vlmxError(VLMXE_IllegalArgument, 
                  "DATA and DEROUTPUT do not have compatible formats.") ;
  }


  if (backMode) {
    derOutput.init(in[IN_DEROUTPUT]) ;
    derOutput.reshape(4) ;
  }

  /* Create output buffers */
  vl::DeviceType deviceType = data.getDeviceType() ;
  vl::DataType dataType = data.getDataType() ;
  vl::TensorShape outputShape = data.getShape() ;
  vl::MexTensor output(context) ;
  vl::MexTensor derData(context) ;

  if (!backMode) {
    output.init(deviceType, dataType, outputShape) ;
  } else {
    if (computeDerData) {
      derData.init(deviceType, dataType, data.getShape()) ;
    }
  }



  if (verbosity > 0) {
    mexPrintf("vl_nnquickrelu: mode %s; %s\n",  
        (data.getDeviceType()==vl::VLDT_GPU)?"gpu":"cpu", "forward") ;
    mexPrintf("vl_nnquickrelu: leak: %f\n", leak) ;
    vl::print("vl_multiboxdetector: output: ", output) ;
    }

  /* -------------------------------------------------------------- */
  /*                                                    Do the work */
  /* -------------------------------------------------------------- */

  vl::ErrorCode error ;
  if (!backMode) {
    error = vl::nnquickrelu_forward(context,
                                    output, 
                                    data, 
                                    leak) ;
  } else {
    error = vl::nnquickrelu_backward(context,
                                    derData,
                                    data,
                                    derOutput,
                                    leak) ;
  }

  /* -------------------------------------------------------------- */
  /*                                                         Finish */
  /* -------------------------------------------------------------- */

  if (error != vl::VLE_Success) {
    mexErrMsgTxt(context.getLastErrorMessage().c_str()) ;
  }
  if (backMode) {
    mxClassID classID ;
    switch (derOutput.getDataType()) {
      case vl::VLDT_Float: classID = mxSINGLE_CLASS ; break ;
      case vl::VLDT_Double: classID = mxDOUBLE_CLASS ; break ;
      default: abort() ;
    }
    out[OUT_RESULT] = (computeDerData) ? derData.relinquish() : 
                                  mxCreateNumericMatrix(0,0,classID,mxREAL) ;
  } else {
    out[OUT_RESULT] = output.relinquish() ;
  }
}
