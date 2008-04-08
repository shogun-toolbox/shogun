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

%sg('send_command', sprintf('set_kernel LINEAR BYTE %f', size_cache));

%sg('set_features', 'TRAIN', int8(traindata_real), 'RAWBYTE');
%sg('send_command', 'init_kernel TRAIN');
%km=sg('get_kernel_matrix');

%sg('set_features', 'TEST', int8(testdata_real), 'RAWBYTE');
%sg('send_command', 'init_kernel TEST');
%km=sg('get_kernel_matrix');


%
% real features;
%

width=1.4;

% CHI2
disp('Chi2');

sg('send_command', sprintf('set_kernel CHI2 REAL %d %f', size_cache, width));

sg('set_features', 'TRAIN', traindata_real);
sg('send_command', 'init_kernel TRAIN');
km=sg('get_kernel_matrix');

sg('set_features', 'TEST', testdata_real);
sg('send_command', 'init_kernel TEST');
km=sg('get_kernel_matrix');


% Const
disp('Const');

c=23;

sg('send_command', sprintf('set_kernel CONST REAL %d %f', size_cache, c));

sg('set_features', 'TRAIN', traindata_real);
sg('send_command', 'init_kernel TRAIN');
km=sg('get_kernel_matrix');

sg('set_features', 'TEST', testdata_real);
sg('send_command', 'init_kernel TEST');
km=sg('get_kernel_matrix');


% Diag
disp('Diag');

diag=23.;

sg('send_command', sprintf('set_kernel DIAG REAL %d %f', size_cache, diag));

sg('set_features', 'TRAIN', traindata_real);
sg('send_command', 'init_kernel TRAIN');
km=sg('get_kernel_matrix');

sg('set_features', 'TEST', testdata_real);
sg('send_command', 'init_kernel TEST');
km=sg('get_kernel_matrix');


% Gaussian
disp('Gaussian');

sg('send_command', sprintf('set_kernel GAUSSIAN REAL %d %f', size_cache, width));

sg('set_features', 'TRAIN', traindata_real);
sg('send_command', 'init_kernel TRAIN');
km=sg('get_kernel_matrix');

sg('set_features', 'TEST', testdata_real);
sg('send_command', 'init_kernel TEST');
km=sg('get_kernel_matrix');


% GaussianShift
disp('GaussianShift');

max_shift=2;
shift_step=1;

sg('send_command', sprintf('set_kernel GAUSSIANSHIFT REAL %d %f %d %d', size_cache, width, max_shift, shift_step));

sg('set_features', 'TRAIN', traindata_real);
sg('send_command', 'init_kernel TRAIN');
km=sg('get_kernel_matrix');

sg('set_features', 'TEST', testdata_real);
sg('send_command', 'init_kernel TEST');
km=sg('get_kernel_matrix');


% Linear
disp('Linear');

scale=1.2;

sg('send_command', sprintf('set_kernel LINEAR REAL %d %f', size_cache, scale));

sg('set_features', 'TRAIN', traindata_real);
sg('send_command', 'init_kernel TRAIN');
km=sg('get_kernel_matrix');

sg('set_features', 'TEST', testdata_real);
sg('send_command', 'init_kernel TEST');
km=sg('get_kernel_matrix');


% Poly
disp('Poly');

degree=4;
inhomogene=0;
use_normalization=1;

sg('send_command', sprintf('set_kernel POLY REAL %d %d %d %d', size_cache, degree, inhomogene, use_normalization));

sg('set_features', 'TRAIN', traindata_real);
sg('send_command', 'init_kernel TRAIN');
km=sg('get_kernel_matrix');

sg('set_features', 'TEST', testdata_real);
sg('send_command', 'init_kernel TEST');
km=sg('get_kernel_matrix');


% sigmoid
disp('Sigmoid');

gamma=1.2;
coef0=1.3;

sg('send_command', sprintf('set_kernel SIGMOID REAL %d %f %f', size_cache, gamma, coef0));

sg('set_features', 'TRAIN', traindata_real);
sg('send_command', 'init_kernel TRAIN');
km=sg('get_kernel_matrix');

sg('set_features', 'TEST', testdata_real);
sg('send_command', 'init_kernel TEST');
km=sg('get_kernel_matrix');


%
% sparse real features
%

% Sparse Gaussian
disp('SparseGaussian');

width=1.3;

sg('send_command', sprintf('set_kernel GAUSSIAN SPARSEREAL %d %f', size_cache, width));

sg('set_features', 'TRAIN', sparse(traindata_real));
sg('send_command', 'init_kernel TRAIN');
km=sg('get_kernel_matrix');

sg('set_features', 'TEST', sparse(testdata_real));
sg('send_command', 'init_kernel TEST');
km=sg('get_kernel_matrix');


% Sparse Linear
disp('SparseLinear');

scale=1.3;

sg('send_command', sprintf('set_kernel LINEAR SPARSEREAL %d %f', size_cache, scale));
sg('set_features', 'TRAIN', sparse(traindata_real));
sg('send_command', 'init_kernel TRAIN');
km=sg('get_kernel_matrix');

sg('set_features', 'TEST', sparse(testdata_real));
sg('send_command', 'init_kernel TEST');
km=sg('get_kernel_matrix');


% Sparse Poly
disp('SparsePoly');

degree=3;
inhomogene=1;
use_normalization=1;

sg('send_command', sprintf('set_kernel POLY SPARSEREAL %d %d %d %d', size_cache, degree, inhomogene, use_normalization));

sg('set_features', 'TRAIN', sparse(traindata_real));
sg('send_command', 'init_kernel TRAIN');
km=sg('get_kernel_matrix');

sg('set_features', 'TEST', sparse(testdata_real));
sg('send_command', 'init_kernel TEST');
km=sg('get_kernel_matrix');


%
% word features
%

maxval=2^16-1;

% LinearWord
disp('LinearWord');

scale=1.4;

sg('send_command', sprintf('set_kernel LINEAR WORD %d %f', size_cache, scale));
sg('set_features', 'TRAIN', uint16(traindata_real*maxval));
sg('send_command', 'init_kernel TRAIN');
km=sg('get_kernel_matrix');

sg('set_features', 'TEST', uint16(testdata_real*maxval));
sg('send_command', 'init_kernel TEST');
km=sg('get_kernel_matrix');


% Poly Match Word
disp('PolyMatchWord');

degree=2;
inhomogene=1;
normalize=1;

sg('send_command', sprintf('set_kernel POLYMATCH WORD %d %d %d %d', size_cache, degree, inhomogene, normalize));

sg('set_features', 'TRAIN', uint16(traindata_real*maxval));
sg('send_command', 'init_kernel TRAIN');
km=sg('get_kernel_matrix');

sg('set_features', 'TEST', uint16(testdata_real*maxval));
sg('send_command', 'init_kernel TEST');
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

sg('send_command', sprintf('set_kernel FIXEDDEGREE CHAR %d %d', size_cache, degree));

sg('set_features', 'TRAIN', traindata_dna, 'DNA');
sg('send_command', 'init_kernel TRAIN');
km=sg('get_kernel_matrix');

sg('set_features', 'TEST', testdata_dna, 'DNA');
sg('send_command', 'init_kernel TEST');
km=sg('get_kernel_matrix');


% Linear String
disp('LinearString');

sg('send_command', sprintf('set_kernel LINEAR CHAR %d', size_cache));

sg('set_features', 'TRAIN', traindata_dna, 'DNA');
sg('send_command', 'init_kernel TRAIN');
km=sg('get_kernel_matrix');

sg('set_features', 'TEST', testdata_dna, 'DNA');
sg('send_command', 'init_kernel TEST');
km=sg('get_kernel_matrix');


% Local Alignment String
disp('LocalAlignmentString');

sg('send_command', sprintf('set_kernel LOCALALIGNMENT CHAR %d', size_cache));

sg('set_features', 'TRAIN', traindata_dna, 'DNA');
sg('send_command', 'init_kernel TRAIN');
km=sg('get_kernel_matrix');

sg('set_features', 'TEST', testdata_dna, 'DNA');
sg('send_command', 'init_kernel TEST');
km=sg('get_kernel_matrix');


% Poly Match String
disp('PolyMatchString');

degree=3;
inhomogene=0;

sg('send_command', sprintf('set_kernel POLYMATCH CHAR %d %d %d', size_cache, degree, inhomogene));

sg('set_features', 'TRAIN', traindata_dna, 'DNA');
sg('send_command', 'init_kernel TRAIN');
km=sg('get_kernel_matrix');

sg('set_features', 'TEST', testdata_dna, 'DNA');
sg('send_command', 'init_kernel TEST');
km=sg('get_kernel_matrix');


% Weighted Degree String
disp('WeightedDegreeString');

degree=20;

sg('send_command', sprintf('set_kernel WEIGHTEDDEGREE CHAR %d %d', size_cache, degree));

sg('set_features', 'TRAIN', traindata_dna, 'DNA');
sg('send_command', 'init_kernel TRAIN');
km=sg('get_kernel_matrix');

sg('set_features', 'TEST', testdata_dna, 'DNA');
sg('send_command', 'init_kernel TEST');
km=sg('get_kernel_matrix');


% Weighted Degree Position String
disp('WeightedDegreePositionString');

degree=20;

sg('send_command', sprintf('set_kernel WEIGHTEDDEGREEPOS CHAR %d %d', size_cache, degree));

sg('set_features', 'TRAIN', traindata_dna, 'DNA');
sg('send_command', 'init_kernel TRAIN');
km=sg('get_kernel_matrix');

sg('set_features', 'TEST', testdata_dna, 'DNA');
sg('send_command', 'init_kernel TEST');
km=sg('get_kernel_matrix');

% Locality Improved String
disp('LocalityImprovedString');

length=5;
inner_degree=5;
outer_degree=inner_degree+2;

sg('send_command', sprintf('set_kernel LIK CHAR %d %d %d %d', size_cache, length, inner_degree, outer_degree));

sg('set_features', 'TRAIN', traindata_dna, 'DNA');
sg('send_command', 'init_kernel TRAIN');
km=sg('get_kernel_matrix');

sg('set_features', 'TEST', testdata_dna, 'DNA');
sg('send_command', 'init_kernel TEST');
km=sg('get_kernel_matrix');

% Simple Locality Improved String
disp('SimpleLocalityImprovedString');

length=5;
inner_degree=5;
outer_degree=inner_degree+2;

sg('send_command', sprintf('set_kernel SLIK CHAR %d %d %d %d', size_cache, length, inner_degree, outer_degree));

sg('set_features', 'TRAIN', traindata_dna, 'DNA');
sg('send_command', 'init_kernel TRAIN');
km=sg('get_kernel_matrix');

sg('set_features', 'TEST', testdata_dna, 'DNA');
sg('send_command', 'init_kernel TEST');
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

sg('send_command', 'add_preproc SORTWORDSTRING');
sg('send_command', sprintf('set_kernel COMMSTRING WORD %d %d %s', size_cache, use_sign, normalization));

sg('set_features', 'TRAIN', traindata_dna, 'DNA');
sg('send_command', sprintf('convert TRAIN STRING CHAR STRING WORD %d %d %d %c', order, order-1, gap, reverse));
sg('send_command', 'attach_preproc TRAIN');
sg('send_command', 'init_kernel TRAIN');
km=sg('get_kernel_matrix');

sg('set_features', 'TEST', testdata_dna, 'DNA');
sg('send_command', sprintf('convert TEST STRING CHAR STRING WORD %d %d %d %c', order, order-1, gap, reverse));
sg('send_command', 'attach_preproc TEST');

sg('send_command', 'init_kernel TEST');
km=sg('get_kernel_matrix');


% Weighted Comm Word String
disp('WeightedCommWordString');

sg('send_command', 'add_preproc SORTWORDSTRING');
sg('send_command', sprintf('set_kernel WEIGHTEDCOMMSTRING WORD %d %d %s', size_cache, use_sign, normalization));

sg('set_features', 'TRAIN', traindata_dna, 'DNA');
sg('send_command', sprintf('convert TRAIN STRING CHAR STRING WORD %d %d %d %c', order, order-1, gap, reverse));
sg('send_command', 'attach_preproc TRAIN');
sg('send_command', 'init_kernel TRAIN');
km=sg('get_kernel_matrix');

sg('set_features', 'TEST', testdata_dna, 'DNA');
sg('send_command', sprintf('convert TEST STRING CHAR STRING WORD %d %d %d %c', order, order-1, gap, reverse));
sg('send_command', 'attach_preproc TEST');

sg('send_command', 'init_kernel TEST');
km=sg('get_kernel_matrix');


% Comm Ulong String
disp('CommUlongString');

sg('send_command', 'add_preproc SORTULONGSTRING');
sg('send_command', sprintf('set_kernel COMMSTRING ULONG %d %d %s', size_cache, use_sign, normalization));

sg('set_features', 'TRAIN', traindata_dna, 'DNA');
sg('send_command', sprintf('convert TRAIN STRING CHAR STRING ULONG %d %d %d %c', order, order-1, gap, reverse));
sg('send_command', 'attach_preproc TRAIN');
sg('send_command', 'init_kernel TRAIN');
km=sg('get_kernel_matrix');

sg('set_features', 'TEST', testdata_dna, 'DNA');
sg('send_command', sprintf('convert TEST STRING CHAR STRING ULONG %d %d %d %c', order, order-1, gap, reverse));
sg('send_command', 'attach_preproc TEST');

sg('send_command', 'init_kernel TEST');
km=sg('get_kernel_matrix');

%
% misc kernels;
%

% Distance
disp('Distance');

width=1.7;

sg('send_command', 'set_distance EUCLIDIAN REAL');
sg('send_command', sprintf('set_kernel DISTANCE %d %f', size_cache, width));

sg('set_features', 'TRAIN', traindata_real);
sg('send_command', 'init_kernel TRAIN');
km=sg('get_kernel_matrix');

sg('set_features', 'TEST', testdata_real);
sg('send_command', 'init_kernel TEST');
km=sg('get_kernel_matrix');


% Combined
disp('Combined');

sg('send_command', sprintf('set_kernel COMBINED %d', size_cache));

sg('send_command', sprintf('add_kernel 1 LINEAR REAL %d', size_cache));
sg('add_features', 'TRAIN', traindata_real);
sg('add_features', 'TEST', testdata_real);

sg('send_command', sprintf('add_kernel 1 GAUSSIAN REAL %d 1', size_cache));
sg('add_features', 'TRAIN', traindata_real);
sg('add_features', 'TEST', testdata_real);

sg('send_command', sprintf('add_kernel 1 POLY REAL %d 3 0', size_cache));
sg('add_features', 'TRAIN', traindata_real);
sg('add_features', 'TEST', testdata_real);

sg('send_command', 'init_kernel TRAIN');
km=sg('get_kernel_matrix');

sg('send_command', 'init_kernel TEST');
km=sg('get_kernel_matrix');


% Plugin Estimate
disp('PluginEstimate w/ HistogramWord');

sg('set_features', 'TRAIN', traindata_dna, 'DNA');
sg('send_command', sprintf('convert TRAIN STRING CHAR STRING WORD %d %d %d %c', order, order-1, gap, reverse));

sg('set_features', 'TEST', testdata_dna, 'DNA');
sg('send_command', sprintf('convert TEST STRING CHAR STRING WORD %d %d %d %c', order, order-1, gap, reverse));

pseudo_pos=1e-1;
pseudo_neg=1e-1;
sg('send_command', sprintf('new_plugin_estimator %f %f', pseudo_pos, pseudo_neg));
sg('set_labels', 'TRAIN', trainlab_dna);
sg('send_command', 'train_estimator');

sg('send_command', sprintf('set_kernel HISTOGRAM WORD %d', size_cache));
sg('send_command', 'init_kernel TRAIN');
km=sg('get_kernel_matrix');

sg('send_command', 'init_kernel TEST');
% not supported yet;
%	lab=sg('send_command', 'plugin_estimate_classify');
km=sg('get_kernel_matrix');

