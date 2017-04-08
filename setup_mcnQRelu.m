function setup_mcnQRelu
%SETUP_MCNQRELU Sets up mcnQRelu by adding its folders to the MATLAB path

root = fileparts(mfilename('fullpath')) ;
addpath(root) ;
addpath(fullfile(root, 'matlab')) ;
addpath(fullfile(root, 'matlab/mex')) ;
