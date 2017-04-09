## QRelu

This module is arguably unnecessary, and exists only for those who 
enjoy chasing extremely marginal-gains.  It provides a small `CUDA`
implementation of the `Leaky ReLU` function. 

### Usage

To use, simply replace calls to `vl_nnrelu()` with calls to 
`vl_nnquickrelu()`.

## Performance

The result of replacing all `vl_nnrelu` calls with `vl_nnquickrelu` in the 
baseline `SSD detector` (implemented [here](https://github.com/albanie/mcnSSD)) 
was an average `6%` improvement in speed across three runs of the Pascal 2007
test set. More results to come.

NOTE: Profiling individual functions is notoriously challenging on the `GPU`, 
so the above should only be used as a very approximate guideline.
