Grave warning: this module contains *untested code*.

## QRelu

This module is arguably unnecessary, and exists only for those who 
enjoy chasing extremely marginal-gains.  It provides a small CUDA 
implementation of the leaky ReLU function. Benchmarks to follow.

### Usage

To use, simply replace calls to `vl_nnrelu()` with calls to 
`vl_nnquickrelu()`.
