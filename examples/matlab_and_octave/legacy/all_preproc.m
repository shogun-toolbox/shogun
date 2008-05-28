% Explicit examples on how to use the different preprocs

num=20;
len=42;
size_cache=10;
width=1.4;
dist=3.3;

%
% real features
%

traindata_real=[randn(2,num)-dist, randn(2,num)+dist, randn(2,num)+dist*[ones(1,num); zeros(1,num)], randn(2,num)+dist*[zeros(1,num); ones(1,num)]];
testdata_real=[randn(2,num+7)-dist, randn(2,num+7)+dist, randn(2,num+7)+dist*[ones(1,num+7); zeros(1,num+7)], randn(2,num+7)+dist*[zeros(1,num+7); ones(1,num+7)]];


% LogPlusOne
disp('LogPlusOne');

sg('add_preproc', 'LOGPLUSONE');
sg('set_kernel', 'CHI2', 'REAL', size_cache, width);

sg('set_features', 'TRAIN', traindata_real);
sg('attach_preproc', 'TRAIN');
sg('init_kernel', 'TRAIN');
km=sg('get_kernel_matrix');

sg('set_features', 'TEST', testdata_real);
sg('attach_preproc', 'TEST');
sg('init_kernel', 'TEST');
km=sg('get_kernel_matrix');

% NormOne
disp('NormOne');

sg('add_preproc', 'NORMONE');
sg('set_kernel', 'CHI2', 'REAL', size_cache, width);

sg('set_features', 'TRAIN', traindata_real);
sg('attach_preproc', 'TRAIN');
sg('init_kernel', 'TRAIN');
km=sg('get_kernel_matrix');

sg('set_features', 'TEST', testdata_real);
sg('attach_preproc', 'TEST');
sg('init_kernel', 'TEST');
km=sg('get_kernel_matrix');


% PruneVarSubMean
disp('PruneVarSubMean');

divide_by_std=1;

sg('add_preproc', 'PRUNEVARSUBMEAN', divide_by_std);
sg('set_kernel', 'CHI2', 'REAL', size_cache, width);

sg('set_features', 'TRAIN', traindata_real);
sg('attach_preproc', 'TRAIN');
sg('init_kernel', 'TRAIN');
km=sg('get_kernel_matrix');

sg('set_features', 'TEST', testdata_real);
sg('attach_preproc', 'TEST');
sg('init_kernel', 'TEST');
km=sg('get_kernel_matrix');

%
% word features;
%

maxval=2^16-1;
traindata_word=uint16(rand(len, num)*maxval);
testdata_word=uint16(rand(len, num)*maxval);

% LinearWord
disp('LinearWord');

scale=1.4;

sg('add_preproc', 'SORTWORD');
sg('set_kernel', 'LINEAR', 'WORD', size_cache, scale);

sg('set_features', 'TRAIN', traindata_word);
sg('attach_preproc', 'TRAIN');
sg('init_kernel', 'TRAIN');
km=sg('get_kernel_matrix');

sg('set_features', 'TEST', testdata_word);
sg('attach_preproc', 'TEST');
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

acgt='ACGT';
trainlab_dna=[ones(1,num/2) -ones(1,num/2)];
traindata_dna=acgt(ceil(4*rand(len,num)));
testdata_dna=acgt(ceil(4*rand(len,num)));

% SortWordString
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


% SortUlongString
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

