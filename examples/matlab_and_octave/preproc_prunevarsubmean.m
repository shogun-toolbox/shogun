size_cache=10;

addpath('tools');
fm_train_real=load_matrix('../data/fm_train_real.dat');
fm_test_real=load_matrix('../data/fm_test_real.dat');


width=1.4;


% PruneVarSubMean
disp('PruneVarSubMean');

divide_by_std=true;

sg('add_preproc', 'PRUNEVARSUBMEAN', divide_by_std);
sg('set_kernel', 'CHI2', 'REAL', size_cache, width);

sg('set_features', 'TRAIN', fm_train_real);
sg('attach_preproc', 'TRAIN');
km=sg('get_kernel_matrix', 'TRAIN');

sg('set_features', 'TEST', fm_test_real);
sg('attach_preproc', 'TEST');
km=sg('get_kernel_matrix', 'TEST');
