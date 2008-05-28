% Explicit examples on how to use the different kernels

size_cache=10;
num=20;
len=33;
dist=2.1;


traindata_real=[randn(2,num)-dist, randn(2,num)+dist, randn(2,num)+dist*[ones(1,num); zeros(1,num)], randn(2,num)+dist*[zeros(1,num); ones(1,num)]];
testdata_real=[randn(2,num+7)-dist, randn(2,num+7)+dist, randn(2,num+7)+dist*[ones(1,num+7); zeros(1,num+7)], randn(2,num+7)+dist*[zeros(1,num+7); ones(1,num+7)]];

%
% byte features
%

% LinearByte is b0rked
disp('LinearByte');

%sg('set_kernel', 'LINEAR', 'BYTE', size_cache);

%sg('set_features', 'TRAIN', int8(traindata_real), 'RAWBYTE');
%sg('init_kernel', 'TRAIN');
%km=sg('get_kernel_matrix');

%sg('set_features', 'TEST', int8(testdata_real), 'RAWBYTE');
%sg('init_kernel', 'TEST');
%km=sg('get_kernel_matrix');


%
% real features;
%

width=1.4;

% CHI2
disp('Chi2');

sg('set_kernel', 'CHI2', 'REAL', size_cache, width);

sg('set_features', 'TRAIN', traindata_real);
sg('init_kernel', 'TRAIN');
km=sg('get_kernel_matrix');

sg('set_features', 'TEST', testdata_real);
sg('init_kernel', 'TEST');
km=sg('get_kernel_matrix');


% Const
disp('Const');

c=23;

sg('set_kernel', 'CONST', 'REAL', size_cache, c);

sg('set_features', 'TRAIN', traindata_real);
sg('init_kernel', 'TRAIN');
km=sg('get_kernel_matrix');

sg('set_features', 'TEST', testdata_real);
sg('init_kernel', 'TEST');
km=sg('get_kernel_matrix');


% Diag
disp('Diag');

diag=23.;

sg('set_kernel', 'DIAG', 'REAL', size_cache, diag);

sg('set_features', 'TRAIN', traindata_real);
sg('init_kernel', 'TRAIN');
km=sg('get_kernel_matrix');

sg('set_features', 'TEST', testdata_real);
sg('init_kernel', 'TEST');
km=sg('get_kernel_matrix');


% Gaussian
disp('Gaussian');

sg('set_kernel', 'GAUSSIAN', 'REAL', size_cache, width);

sg('set_features', 'TRAIN', traindata_real);
sg('init_kernel', 'TRAIN');
km=sg('get_kernel_matrix');

sg('set_features', 'TEST', testdata_real);
sg('init_kernel', 'TEST');
km=sg('get_kernel_matrix');


% GaussianShift
disp('GaussianShift');

max_shift=2;
shift_step=1;

sg('set_kernel', 'GAUSSIANSHIFT', 'REAL', size_cache, width, max_shift, shift_step);

sg('set_features', 'TRAIN', traindata_real);
sg('init_kernel', 'TRAIN');
km=sg('get_kernel_matrix');

sg('set_features', 'TEST', testdata_real);
sg('init_kernel', 'TEST');
km=sg('get_kernel_matrix');


% Linear
disp('Linear');

scale=1.2;

sg('set_kernel', 'LINEAR', 'REAL', size_cache, scale);

sg('set_features', 'TRAIN', traindata_real);
sg('init_kernel', 'TRAIN');
km=sg('get_kernel_matrix');

sg('set_features', 'TEST', testdata_real);
sg('init_kernel', 'TEST');
km=sg('get_kernel_matrix');


% Poly
disp('Poly');

degree=4;
inhomogene=0;
use_normalization=1;

sg('set_kernel', 'POLY', 'REAL', size_cache, degree, inhomogene, use_normalization);

sg('set_features', 'TRAIN', traindata_real);
sg('init_kernel', 'TRAIN');
km=sg('get_kernel_matrix');

sg('set_features', 'TEST', testdata_real);
sg('init_kernel', 'TEST');
km=sg('get_kernel_matrix');


% sigmoid
disp('Sigmoid');

gamma=1.2;
coef0=1.3;

sg('set_kernel', 'SIGMOID', 'REAL', size_cache, gamma, coef0);

sg('set_features', 'TRAIN', traindata_real);
sg('init_kernel', 'TRAIN');
km=sg('get_kernel_matrix');

sg('set_features', 'TEST', testdata_real);
sg('init_kernel', 'TEST');
km=sg('get_kernel_matrix');


%
% sparse real features
%

% Sparse Gaussian
disp('SparseGaussian');

width=1.3;

sg('set_kernel', 'GAUSSIAN', 'SPARSEREAL', size_cache, width);

sg('set_features', 'TRAIN', sparse(traindata_real));
sg('init_kernel', 'TRAIN');
km=sg('get_kernel_matrix');

sg('set_features', 'TEST', sparse(testdata_real));
sg('init_kernel', 'TEST');
km=sg('get_kernel_matrix');


% Sparse Linear
disp('SparseLinear');

scale=1.3;

sg('set_kernel', 'LINEAR', 'SPARSEREAL', size_cache, scale);

sg('set_features', 'TRAIN', sparse(traindata_real));
sg('init_kernel', 'TRAIN');
km=sg('get_kernel_matrix');

sg('set_features', 'TEST', sparse(testdata_real));
sg('init_kernel', 'TEST');
km=sg('get_kernel_matrix');


% Sparse Poly
disp('SparsePoly');

degree=3;
inhomogene=1;
use_normalization=1;

sg('set_kernel', 'POLY', 'SPARSEREAL', size_cache, degree, inhomogene, use_normalization);

sg('set_features', 'TRAIN', sparse(traindata_real));
sg('init_kernel', 'TRAIN');
km=sg('get_kernel_matrix');

sg('set_features', 'TEST', sparse(testdata_real));
sg('init_kernel', 'TEST');
km=sg('get_kernel_matrix');


%
% word features
%

maxval=2^16-1;

% LinearWord
disp('LinearWord');

scale=1.4;

sg('set_kernel', 'LINEAR', 'WORD', size_cache, scale);

sg('set_features', 'TRAIN', uint16(traindata_real*maxval));
sg('init_kernel', 'TRAIN');
km=sg('get_kernel_matrix');

sg('set_features', 'TEST', uint16(testdata_real*maxval));
sg('init_kernel', 'TEST');
km=sg('get_kernel_matrix');


% Poly Match Word
disp('PolyMatchWord');

degree=2;
inhomogene=1;
normalize=1;

sg('set_kernel', 'POLYMATCH', 'WORD', size_cache, degree, inhomogene, normalize);

sg('set_features', 'TRAIN', uint16(traindata_real*maxval));
sg('init_kernel', 'TRAIN');
km=sg('get_kernel_matrix');

sg('set_features', 'TEST', uint16(testdata_real*maxval));
sg('init_kernel', 'TEST');
km=sg('get_kernel_matrix');

%
% string features
%

acgt='ACGT';
trainlab_dna=[ones(1,num/2) -ones(1,num/2)];
traindata_dna=acgt(ceil(4*rand(len,num)));
testdata_dna=acgt(ceil(4*rand(len,num)));

% Fixed Degree String
disp('FixedDegreeString');

degree=3;

sg('set_kernel', 'FIXEDDEGREE', 'CHAR', size_cache, degree);

sg('set_features', 'TRAIN', traindata_dna, 'DNA');
sg('init_kernel', 'TRAIN');
km=sg('get_kernel_matrix');

sg('set_features', 'TEST', testdata_dna, 'DNA');
sg('init_kernel', 'TEST');
km=sg('get_kernel_matrix');


% Linear String
disp('LinearString');

sg('set_kernel', 'LINEAR', 'CHAR', size_cache);

sg('set_features', 'TRAIN', traindata_dna, 'DNA');
sg('init_kernel', 'TRAIN');
km=sg('get_kernel_matrix');

sg('set_features', 'TEST', testdata_dna, 'DNA');
sg('init_kernel', 'TEST');
km=sg('get_kernel_matrix');


% Local Alignment String
disp('LocalAlignmentString');

sg('set_kernel', 'LOCALALIGNMENT', 'CHAR', size_cache);

sg('set_features', 'TRAIN', traindata_dna, 'DNA');
sg('init_kernel', 'TRAIN');
km=sg('get_kernel_matrix');

sg('set_features', 'TEST', testdata_dna, 'DNA');
sg('init_kernel', 'TEST');
km=sg('get_kernel_matrix');


% Poly Match String
disp('PolyMatchString');

degree=3;
inhomogene=0;

sg('set_kernel', 'POLYMATCH', 'CHAR', size_cache, degree, inhomogene);

sg('set_features', 'TRAIN', traindata_dna, 'DNA');
sg('init_kernel', 'TRAIN');
km=sg('get_kernel_matrix');

sg('set_features', 'TEST', testdata_dna, 'DNA');
sg('init_kernel', 'TEST');
km=sg('get_kernel_matrix');


% Weighted Degree String
disp('WeightedDegreeString');

degree=20;

sg('set_kernel', 'WEIGHTEDDEGREE', 'CHAR', size_cache, degree);

sg('set_features', 'TRAIN', traindata_dna, 'DNA');
sg('init_kernel', 'TRAIN');
km=sg('get_kernel_matrix');

sg('set_features', 'TEST', testdata_dna, 'DNA');
sg('init_kernel', 'TEST');
km=sg('get_kernel_matrix');


% Weighted Degree Position String
disp('WeightedDegreePositionString');

degree=20;

sg('set_kernel', 'WEIGHTEDDEGREEPOS', 'CHAR', size_cache, degree);

sg('set_features', 'TRAIN', traindata_dna, 'DNA');
sg('init_kernel', 'TRAIN');
km=sg('get_kernel_matrix');

sg('set_features', 'TEST', testdata_dna, 'DNA');
sg('init_kernel', 'TEST');
km=sg('get_kernel_matrix');

% Locality Improved String
disp('LocalityImprovedString');

length=5;
inner_degree=5;
outer_degree=inner_degree+2;

sg('set_kernel', 'LIK', 'CHAR', size_cache, length, inner_degree, outer_degree);

sg('set_features', 'TRAIN', traindata_dna, 'DNA');
sg('init_kernel', 'TRAIN');
km=sg('get_kernel_matrix');

sg('set_features', 'TEST', testdata_dna, 'DNA');
sg('init_kernel', 'TEST');
km=sg('get_kernel_matrix');

% Simple Locality Improved String
disp('SimpleLocalityImprovedString');

length=5;
inner_degree=5;
outer_degree=inner_degree+2;

sg('set_kernel', 'SLIK', 'CHAR', size_cache, length, inner_degree, outer_degree);

sg('set_features', 'TRAIN', traindata_dna, 'DNA');
sg('init_kernel', 'TRAIN');
km=sg('get_kernel_matrix');

sg('set_features', 'TEST', testdata_dna, 'DNA');
sg('init_kernel', 'TEST');
km=sg('get_kernel_matrix');


%
% complex string features;
%

order=3;
gap=0;
reverse='n'; % bit silly to not use boolean, set 'r' to yield true
use_sign=0;
normalization='FULL';

% Comm Word String
disp('CommWordString');

sg('add_preproc', 'SORTWORDSTRING');
sg('set_kernel', 'COMMSTRING', 'WORD', size_cache, use_sign, normalization);

sg('set_features', 'TRAIN', traindata_dna, 'DNA');
sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse);
sg('attach_preproc', 'TRAIN');
sg('init_kernel', 'TRAIN');
km=sg('get_kernel_matrix');

sg('set_features', 'TEST', testdata_dna, 'DNA');
sg('convert', 'TEST', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse);
sg('attach_preproc', 'TEST');
sg('init_kernel', 'TEST');
km=sg('get_kernel_matrix');


% Weighted Comm Word String
disp('WeightedCommWordString');

sg('add_preproc', 'SORTWORDSTRING');
sg('set_kernel', 'WEIGHTEDCOMMSTRING', 'WORD', size_cache, use_sign, normalization);

sg('set_features', 'TRAIN', traindata_dna, 'DNA');
sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse);
sg('attach_preproc', 'TRAIN');
sg('init_kernel', 'TRAIN');
km=sg('get_kernel_matrix');

sg('set_features', 'TEST', testdata_dna, 'DNA');
sg('convert', 'TEST', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse);
sg('attach_preproc', 'TEST');
sg('init_kernel', 'TEST');
km=sg('get_kernel_matrix');


% Comm Ulong String
disp('CommUlongString');

sg('add_preproc', 'SORTULONGSTRING');
sg('set_kernel', 'COMMSTRING', 'ULONG', size_cache, use_sign, normalization);

sg('set_features', 'TRAIN', traindata_dna, 'DNA');
sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', 'ULONG', order, order-1, gap, reverse);
sg('attach_preproc', 'TRAIN');
sg('init_kernel', 'TRAIN');
km=sg('get_kernel_matrix');

sg('set_features', 'TEST', testdata_dna, 'DNA');
sg('convert', 'TEST', 'STRING', 'CHAR', 'STRING', 'ULONG', order, order-1, gap, reverse);
sg('attach_preproc', 'TEST');
sg('init_kernel', 'TEST');
km=sg('get_kernel_matrix');

%
% misc kernels;
%

% Distance
disp('Distance');

width=1.7;

sg('set_distance', 'EUCLIDIAN', 'REAL');
sg('set_kernel', 'DISTANCE', size_cache, width);

sg('set_features', 'TRAIN', traindata_real);
sg('init_kernel', 'TRAIN');
km=sg('get_kernel_matrix');

sg('set_features', 'TEST', testdata_real);
sg('init_kernel', 'TEST');
km=sg('get_kernel_matrix');


% Combined
disp('Combined');

sg('set_kernel', 'COMBINED', size_cache);

sg('add_kernel', 1, 'LINEAR', 'REAL', size_cache);
sg('add_features', 'TRAIN', traindata_real);
sg('add_features', 'TEST', testdata_real);

sg('add_kernel', 1, 'GAUSSIAN', 'REAL', size_cache, 1);
sg('add_features', 'TRAIN', traindata_real);
sg('add_features', 'TEST', testdata_real);

sg('add_kernel', 1, 'POLY', 'REAL', size_cache, 3, 0);
sg('add_features', 'TRAIN', traindata_real);
sg('add_features', 'TEST', testdata_real);

sg('init_kernel', 'TRAIN');
km=sg('get_kernel_matrix');

sg('init_kernel', 'TEST');
km=sg('get_kernel_matrix');


% Plugin Estimate
disp('PluginEstimate w/ HistogramWord');

sg('set_features', 'TRAIN', traindata_dna, 'DNA');
sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse);

sg('set_features', 'TEST', testdata_dna, 'DNA');
sg('convert', 'TEST', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse);

pseudo_pos=1e-1;
pseudo_neg=1e-1;
sg('new_plugin_estimator', pseudo_pos, pseudo_neg);
sg('set_labels', 'TRAIN', trainlab_dna);
sg('train_estimator');

sg('set_kernel', 'HISTOGRAM', 'WORD', size_cache);
sg('init_kernel', 'TRAIN');
km=sg('get_kernel_matrix');

sg('init_kernel', 'TEST');
% not supported yet;
%	lab=sg('plugin_estimate_classify');
km=sg('get_kernel_matrix');

