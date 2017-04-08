%VL_NNQUICKRELU CNN marginally more efficient rectified linear unit.
%   Y = VL_NNQUICKRELU(X) applies the rectified linear unit to the data
%   X. X can have arbitrary size.
%
%   DZDX = VL_NNQUICKRELU(X, DZDY) computes the derivative of the block
%   projected onto DZDY. DZDX and DZDY have the same dimensions as
%   X and Y respectively.
%
%   VL_NNQUICKRELU(...,'OPT',VALUE,...) takes the following options:
%
%   `Leak`:: 0
%      Set the leak factor, a non-negative number. Y is equal to X if
%      X is not smaller than zero; otherwise, Y is equal to X
%      multipied by the leak factor. By default, the leak factor is
%      zero; for values greater than that one obtains the leaky ReLU
%      unit.
%
%   ADVANCED USAGE
%
%   As a further optimization, in the backward computation it is
%   possible to replace X with Y, namely, if Y = VL_NNQUICKRELU(X), then
%   VL_NNQUICKRELU(X,DZDY) gives the same result as VL_NNQUICKRELU(Y,DZDY).
%   This is useful because it means that the buffer X does not need to
%   be remembered in the backward pass.
