% Explicit examples on how to use the different kernels

size_cache=10;

addpath('tools');
fm_train_real=load_matrix('../data/fm_train_real.dat');
fm_test_real=load_matrix('../data/fm_test_real.dat');
label_train_dna=load_matrix('../data/label_train_dna.dat');
fm_train_dna=load_matrix('../data/fm_train_dna.dat');
fm_test_dna=load_matrix('../data/fm_test_dna.dat');
fm_train_word=uint16(load_matrix('../data/fm_train_word.dat'));
fm_test_word=uint16(load_matrix('../data/fm_test_word.dat'));
fm_train_byte=uint8(load_matrix('../data/fm_train_byte.dat'));
fm_test_byte=uint8(load_matrix('../data/fm_test_byte.dat'));


%
% byte features
%

% LinearByte is b0rked
disp('LinearByte');

%sg('set_kernel', 'LINEAR', 'BYTE', size_cache);

%sg('set_features', 'TRAIN', fm_train_byte, 'RAWBYTE');
%km=sg('get_kernel_matrix', 'TRAIN');

%sg('set_features', 'TEST', fm_test_byte, 'RAWBYTE');
%km=sg('get_kernel_matrix', 'TEST');


%
% real features;
%

width=1.4;

% CHI2
disp('Chi2');

sg('set_kernel', 'CHI2', 'REAL', size_cache, width);

sg('set_features', 'TRAIN', fm_train_real);
km=sg('get_kernel_matrix', 'TRAIN');

sg('set_features', 'TEST', fm_test_real);
km=sg('get_kernel_matrix', 'TEST');


% Const
disp('Const');

c=23;

sg('set_kernel', 'CONST', 'REAL', size_cache, c);

sg('set_features', 'TRAIN', fm_train_real);
km=sg('get_kernel_matrix', 'TRAIN');

sg('set_features', 'TEST', fm_test_real);
km=sg('get_kernel_matrix', 'TEST');


% Diag
disp('Diag');

diag=23.;

sg('set_kernel', 'DIAG', 'REAL', size_cache, diag);

sg('set_features', 'TRAIN', fm_train_real);
km=sg('get_kernel_matrix', 'TRAIN');

sg('set_features', 'TEST', fm_test_real);
km=sg('get_kernel_matrix', 'TEST');


% Gaussian
disp('Gaussian');

sg('set_kernel', 'GAUSSIAN', 'REAL', size_cache, width);

sg('set_features', 'TRAIN', fm_train_real);
km=sg('get_kernel_matrix', 'TEST');

sg('set_features', 'TEST', fm_test_real);
km=sg('get_kernel_matrix', 'TRAIN');


% GaussianShift
disp('GaussianShift');

max_shift=2;
shift_step=1;

sg('set_kernel', 'GAUSSIANSHIFT', 'REAL', size_cache, width, max_shift, shift_step);

sg('set_features', 'TRAIN', fm_train_real);
km=sg('get_kernel_matrix', 'TRAIN');

sg('set_features', 'TEST', fm_test_real);
km=sg('get_kernel_matrix', 'TEST');


% Linear
disp('Linear');

scale=1.2;

sg('set_kernel', 'LINEAR', 'REAL', size_cache, scale);

sg('set_features', 'TRAIN', fm_train_real);
km=sg('get_kernel_matrix', 'TRAIN');

sg('set_features', 'TEST', fm_test_real);
km=sg('get_kernel_matrix', 'TEST');


% Poly
disp('Poly');

degree=4;
inhomogene=false;
use_normalization=true;

sg('set_kernel', 'POLY', 'REAL', size_cache, degree, inhomogene, use_normalization);

sg('set_features', 'TRAIN', fm_train_real);
km=sg('get_kernel_matrix', 'TRAIN');

sg('set_features', 'TEST', fm_test_real);
km=sg('get_kernel_matrix', 'TEST');


% sigmoid
disp('Sigmoid');

gamma=1.2;
coef0=1.3;

sg('set_kernel', 'SIGMOID', 'REAL', size_cache, gamma, coef0);

sg('set_features', 'TRAIN', fm_train_real);
km=sg('get_kernel_matrix', 'TRAIN');

sg('set_features', 'TEST', fm_test_real);
km=sg('get_kernel_matrix', 'TEST');


%
% sparse real features
%

% Sparse Gaussian
disp('SparseGaussian');

width=1.3;

sg('set_kernel', 'GAUSSIAN', 'SPARSEREAL', size_cache, width);

sg('set_features', 'TRAIN', sparse(fm_train_real));
km=sg('get_kernel_matrix', 'TRAIN');

sg('set_features', 'TEST', sparse(fm_test_real));
km=sg('get_kernel_matrix', 'TEST');


% Sparse Linear
disp('SparseLinear');

scale=1.3;

sg('set_kernel', 'LINEAR', 'SPARSEREAL', size_cache, scale);

sg('set_features', 'TRAIN', sparse(fm_train_real));
km=sg('get_kernel_matrix', 'TRAIN');

sg('set_features', 'TEST', sparse(fm_test_real));
km=sg('get_kernel_matrix', 'TEST');


% Sparse Poly
disp('SparsePoly');

degree=3;
inhomogene=true;
use_normalization=false;

sg('set_kernel', 'POLY', 'SPARSEREAL', size_cache, degree, inhomogene, use_normalization);

sg('set_features', 'TRAIN', sparse(fm_train_real));
km=sg('get_kernel_matrix', 'TRAIN');

sg('set_features', 'TEST', sparse(fm_test_real));
km=sg('get_kernel_matrix', 'TEST');


%
% word features
%

% LinearWord
disp('LinearWord');

scale=1.4;

sg('set_kernel', 'LINEAR', 'WORD', size_cache, scale);

sg('set_features', 'TRAIN', fm_train_word);
km=sg('get_kernel_matrix', 'TRAIN');

sg('set_features', 'TEST', fm_test_word);
km=sg('get_kernel_matrix', 'TEST');

%
% string features
%


% Fixed Degree String
disp('FixedDegreeString');

degree=3;

sg('set_kernel', 'FIXEDDEGREE', 'CHAR', size_cache, degree);

sg('set_features', 'TRAIN', fm_train_dna, 'DNA');
km=sg('get_kernel_matrix', 'TRAIN');

sg('set_features', 'TEST', fm_test_dna, 'DNA');
km=sg('get_kernel_matrix', 'TEST');


% Linear String
disp('LinearString');

sg('set_kernel', 'LINEAR', 'CHAR', size_cache);

sg('set_features', 'TRAIN', fm_train_dna, 'DNA');
km=sg('get_kernel_matrix', 'TRAIN');

sg('set_features', 'TEST', fm_test_dna, 'DNA');
km=sg('get_kernel_matrix', 'TEST');


% Local Alignment String
disp('LocalAlignmentString');

sg('set_kernel', 'LOCALALIGNMENT', 'CHAR', size_cache);

sg('set_features', 'TRAIN', fm_train_dna, 'DNA');
km=sg('get_kernel_matrix', 'TRAIN');

sg('set_features', 'TEST', fm_test_dna, 'DNA');
km=sg('get_kernel_matrix', 'TEST');

% Oligo String
k=3;
w=1.2;

sg('set_kernel', 'OLIGO', 'CHAR', size_cache, k, w);

sg('set_features', 'TRAIN', fm_train_dna, 'DNA');
km=sg('get_kernel_matrix', 'TRAIN');

sg('set_features', 'TEST', fm_test_dna, 'DNA');
km=sg('get_kernel_matrix', 'TEST');

% Poly Match String
disp('PolyMatchString');

degree=3;
inhomogene=false;

sg('set_kernel', 'POLYMATCH', 'CHAR', size_cache, degree, inhomogene);

sg('set_features', 'TRAIN', fm_train_dna, 'DNA');
km=sg('get_kernel_matrix', 'TRAIN');

sg('set_features', 'TEST', fm_test_dna, 'DNA');
km=sg('get_kernel_matrix', 'TEST');


% Weighted Degree String
disp('WeightedDegreeString');

degree=20;

sg('set_kernel', 'WEIGHTEDDEGREE', 'CHAR', size_cache, degree);

sg('set_features', 'TRAIN', fm_train_dna, 'DNA');
km=sg('get_kernel_matrix', 'TRAIN');

sg('set_features', 'TEST', fm_test_dna, 'DNA');
km=sg('get_kernel_matrix', 'TEST');


% Weighted Degree Position String
disp('WeightedDegreePositionString');

degree=20;

sg('set_kernel', 'WEIGHTEDDEGREEPOS', 'CHAR', size_cache, degree);

sg('set_features', 'TRAIN', fm_train_dna, 'DNA');
km=sg('get_kernel_matrix', 'TRAIN');

sg('set_features', 'TEST', fm_test_dna, 'DNA');
km=sg('get_kernel_matrix', 'TEST');

% Locality Improved String
disp('LocalityImprovedString');

length=5;
inner_degree=5;
outer_degree=inner_degree+2;

sg('set_kernel', 'LIK', 'CHAR', size_cache, length, inner_degree, outer_degree);

sg('set_features', 'TRAIN', fm_train_dna, 'DNA');
km=sg('get_kernel_matrix', 'TRAIN');

sg('set_features', 'TEST', fm_test_dna, 'DNA');
km=sg('get_kernel_matrix', 'TEST');

% Simple Locality Improved String
disp('SimpleLocalityImprovedString');

length=5;
inner_degree=5;
outer_degree=inner_degree+2;

sg('set_kernel', 'SLIK', 'CHAR', size_cache, length, inner_degree, outer_degree);

sg('set_features', 'TRAIN', fm_train_dna, 'DNA');
km=sg('get_kernel_matrix', 'TRAIN');

sg('set_features', 'TEST', fm_test_dna, 'DNA');
km=sg('get_kernel_matrix', 'TEST');


%
% complex string features;
%

order=3;
gap=0;
reverse='n'; % bit silly to not use boolean, set 'r' to yield true
use_sign=false;
normalization='FULL';


% Poly Match WordString
disp('PolyMatchWordString');

degree=2;
inhomogene=true;

sg('add_preproc', 'SORTWORDSTRING');
sg('set_kernel', 'POLYMATCH', 'WORD', size_cache, degree, inhomogene);

sg('set_features', 'TRAIN', fm_train_dna, 'DNA');
sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse);
sg('attach_preproc', 'TRAIN');
km=sg('get_kernel_matrix', 'TRAIN');

sg('set_features', 'TEST', fm_test_dna, 'DNA');
sg('convert', 'TEST', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse);
sg('attach_preproc', 'TEST');
km=sg('get_kernel_matrix', 'TEST');


% Comm Word String
disp('CommWordString');

sg('add_preproc', 'SORTWORDSTRING');
sg('set_kernel', 'COMMSTRING', 'WORD', size_cache, use_sign, normalization);

sg('set_features', 'TRAIN', fm_train_dna, 'DNA');
sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse);
sg('attach_preproc', 'TRAIN');
km=sg('get_kernel_matrix', 'TRAIN');

sg('set_features', 'TEST', fm_test_dna, 'DNA');
sg('convert', 'TEST', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse);
sg('attach_preproc', 'TEST');
km=sg('get_kernel_matrix', 'TEST');


% Weighted Comm Word String
disp('WeightedCommWordString');

sg('add_preproc', 'SORTWORDSTRING');
sg('set_kernel', 'WEIGHTEDCOMMSTRING', 'WORD', size_cache, use_sign, normalization);

sg('set_features', 'TRAIN', fm_train_dna, 'DNA');
sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse);
sg('attach_preproc', 'TRAIN');
km=sg('get_kernel_matrix', 'TRAIN');

sg('set_features', 'TEST', fm_test_dna, 'DNA');
sg('convert', 'TEST', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse);
sg('attach_preproc', 'TEST');
km=sg('get_kernel_matrix', 'TEST');


% Comm Ulong String
disp('CommUlongString');

sg('add_preproc', 'SORTULONGSTRING');
sg('set_kernel', 'COMMSTRING', 'ULONG', size_cache, use_sign, normalization);

sg('set_features', 'TRAIN', fm_train_dna, 'DNA');
sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', 'ULONG', order, order-1, gap, reverse);
sg('attach_preproc', 'TRAIN');
km=sg('get_kernel_matrix', 'TRAIN');

sg('set_features', 'TEST', fm_test_dna, 'DNA');
sg('convert', 'TEST', 'STRING', 'CHAR', 'STRING', 'ULONG', order, order-1, gap, reverse);
sg('attach_preproc', 'TEST');
km=sg('get_kernel_matrix', 'TEST');

%
% misc kernels;
%

% Distance
disp('Distance');

width=1.7;

sg('set_distance', 'EUCLIDIAN', 'REAL');
sg('set_kernel', 'DISTANCE', size_cache, width);

sg('set_features', 'TRAIN', fm_train_real);
km=sg('get_kernel_matrix', 'TRAIN');

sg('set_features', 'TEST', fm_test_real);
km=sg('get_kernel_matrix', 'TEST');


% Combined
disp('Combined');

sg('clean_features','TRAIN');
sg('clean_features','TEST');

sg('set_kernel', 'COMBINED', size_cache);

sg('add_kernel', 1, 'LINEAR', 'REAL', size_cache);
sg('add_features', 'TRAIN', fm_train_real);
sg('add_features', 'TEST', fm_test_real);

sg('add_kernel', 1, 'GAUSSIAN', 'REAL', size_cache, 1);
sg('add_features', 'TRAIN', fm_train_real);
sg('add_features', 'TEST', fm_test_real);

sg('add_kernel', 1, 'POLY', 'REAL', size_cache, 3, false);
sg('add_features', 'TRAIN', fm_train_real);
sg('add_features', 'TEST', fm_test_real);

km=sg('get_kernel_matrix', 'TRAIN');

km=sg('get_kernel_matrix', 'TEST');


% Plugin Estimate
disp('PluginEstimate w/ HistogramWord');

sg('set_features', 'TRAIN', fm_train_dna, 'DNA');
sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse);

sg('set_features', 'TEST', fm_test_dna, 'DNA');
sg('convert', 'TEST', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse);

pseudo_pos=1e-1;
pseudo_neg=1e-1;
sg('new_plugin_estimator', pseudo_pos, pseudo_neg);
sg('set_labels', 'TRAIN', label_train_dna);
sg('train_estimator');

sg('set_kernel', 'HISTOGRAM', 'WORD', size_cache);
km=sg('get_kernel_matrix', 'TRAIN');

% not supported yet;
%	lab=sg('plugin_estimate_classify');
km=sg('get_kernel_matrix', 'TEST');

