function test_QRelu
% --------------------------
% run tests for QRelu module
% --------------------------

% add tests to path
addpath(fullfile(fileparts(mfilename('fullpath')), 'matlab/xtest')) ;
addpath(fullfile(vl_rootnn, 'matlab/xtest/suite')) ;

% test network layers
run_quickrelu_tests('command', 'nn') ;
