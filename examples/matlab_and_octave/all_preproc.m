% Explicit examples on how to use the different preprocs

size_cache=10;

addpath('tools');
fm_train_real=load_matrix('../data/fm_train_real.dat');
fm_test_real=load_matrix('../data/fm_test_real.dat');
fm_train_word=load_matrix('../data/fm_train_word.dat');
fm_test_word=load_matrix('../data/fm_test_word.dat');
fm_train_dna=load_matrix('../data/fm_train_dna.dat');
fm_test_dna=load_matrix('../data/fm_test_dna.dat');


%
% real features
%

width=1.4;

% LogPlusOne
disp('LogPlusOne');

sg('add_preproc', 'LOGPLUSONE');
sg('set_kernel', 'CHI2', 'REAL', size_cache, width);

sg('set_features', 'TRAIN', fm_train_real);
sg('attach_preproc', 'TRAIN');
sg('init_kernel', 'TRAIN');
km=sg('get_kernel_matrix');

sg('set_features', 'TEST', fm_test_real);
sg('attach_preproc', 'TEST');
sg('init_kernel', 'TEST');
km=sg('get_kernel_matrix');

% NormOne
disp('NormOne');

sg('add_preproc', 'NORMONE');
sg('set_kernel', 'CHI2', 'REAL', size_cache, width);

sg('set_features', 'TRAIN', fm_train_real);
sg('attach_preproc', 'TRAIN');
sg('init_kernel', 'TRAIN');
km=sg('get_kernel_matrix');

sg('set_features', 'TEST', fm_test_real);
sg('attach_preproc', 'TEST');
sg('init_kernel', 'TEST');
km=sg('get_kernel_matrix');


% PruneVarSubMean
disp('PruneVarSubMean');

divide_by_std=true;

sg('add_preproc', 'PRUNEVARSUBMEAN', divide_by_std);
sg('set_kernel', 'CHI2', 'REAL', size_cache, width);

sg('set_features', 'TRAIN', fm_train_real);
sg('attach_preproc', 'TRAIN');
sg('init_kernel', 'TRAIN');
km=sg('get_kernel_matrix');

sg('set_features', 'TEST', fm_test_real);
sg('attach_preproc', 'TEST');
sg('init_kernel', 'TEST');
km=sg('get_kernel_matrix');


%
% complex string features;
%

order=3;
gap=0;
reverse='n'; % bit silly to not use boolean, set 'r' to yield true
use_sign=false;
normalization='FULL';


% SortWordString
disp('CommWordString');

sg('add_preproc', 'SORTWORDSTRING');
sg('set_kernel', 'COMMSTRING', 'WORD', size_cache, use_sign, normalization);

sg('set_features', 'TRAIN', fm_train_dna, 'DNA');
sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse);
sg('attach_preproc', 'TRAIN');
sg('init_kernel', 'TRAIN');
km=sg('get_kernel_matrix');

sg('set_features', 'TEST', fm_test_dna, 'DNA');
sg('convert', 'TEST', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse);
sg('attach_preproc', 'TEST');
sg('init_kernel', 'TEST');
km=sg('get_kernel_matrix');


% SortUlongString
disp('CommUlongString');

sg('add_preproc', 'SORTULONGSTRING');
sg('set_kernel', 'COMMSTRING', 'ULONG', size_cache, use_sign, normalization);

sg('set_features', 'TRAIN', fm_train_dna, 'DNA');
sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', 'ULONG', order, order-1, gap, reverse);
sg('attach_preproc', 'TRAIN');
sg('init_kernel', 'TRAIN');
km=sg('get_kernel_matrix');

sg('set_features', 'TEST', fm_test_dna, 'DNA');
sg('convert', 'TEST', 'STRING', 'CHAR', 'STRING', 'ULONG', order, order-1, gap, reverse);
sg('attach_preproc', 'TEST');
sg('init_kernel', 'TEST');
km=sg('get_kernel_matrix');

