init_shogun

% Explicit examples on how to use the different preprocs

addpath('tools');
fm_train_real=load_matrix('../data/fm_train_real.dat');
fm_test_real=load_matrix('../data/fm_test_real.dat');
fm_train_word=load_matrix('../data/fm_train_word.dat');
fm_test_word=load_matrix('../data/fm_test_word.dat');
fm_train_dna=load_matrix('../data/fm_train_dna.dat');
fm_test_dna=load_matrix('../data/fm_test_dna.dat');


%LogPlusOne
disp('LogPlusOne')

feats_train=RealFeatures(fm_train_real);
feats_test=RealFeatures(fm_test_real);

preproc=LogPlusOne();
preproc.init(feats_train);
feats_train.add_preproc(preproc);
feats_train.apply_preproc();
feats_test.add_preproc(preproc);
feats_test.apply_preproc();

width=1.4;
size_cache=10;

kernel=Chi2Kernel(feats_train, feats_train, width, size_cache);

km_train=kernel.get_kernel_matrix();
kernel.init(feats_train, feats_test);
km_test=kernel.get_kernel_matrix();

%NormOne
disp('NormOne')

feats_train=RealFeatures(fm_train_real);
feats_test=RealFeatures(fm_test_real);

preproc=NormOne();
preproc.init(feats_train);
feats_train.add_preproc(preproc);
feats_train.apply_preproc();
feats_test.add_preproc(preproc);
feats_test.apply_preproc();

width=1.4;
size_cache=10;

kernel=Chi2Kernel(feats_train, feats_train, width, size_cache);

km_train=kernel.get_kernel_matrix();
kernel.init(feats_train, feats_test);
km_test=kernel.get_kernel_matrix();

%PruneVarSubMean
disp('PruneVarSubMean')

feats_train=RealFeatures(fm_train_real);
feats_test=RealFeatures(fm_test_real);

preproc=PruneVarSubMean();
preproc.init(feats_train);
feats_train.add_preproc(preproc);
feats_train.apply_preproc();
feats_test.add_preproc(preproc);
feats_test.apply_preproc();

width=1.4;
size_cache=10;

kernel=Chi2Kernel(feats_train, feats_train, width, size_cache);

km_train=kernel.get_kernel_matrix();
kernel.init(feats_train, feats_test);
km_test=kernel.get_kernel_matrix();

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% complex string features
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%CommWordString
disp('CommWordString')

order=3;
gap=0;
reverse=false;

charfeat=StringCharFeatures(DNA);
charfeat.set_string_features(fm_train_dna);
feats_train=StringWordFeatures(charfeat.get_alphabet());
feats_train.obtain_from_char(charfeat, order-1, order, gap, reverse);
preproc=SortWordString();
preproc.init(feats_train);
feats_train.add_preproc(preproc);
feats_train.apply_preproc();

charfeat=StringCharFeatures(DNA);
charfeat.set_string_features(fm_test_dna);
feats_test=StringWordFeatures(charfeat.get_alphabet());
feats_test.obtain_from_char(charfeat, order-1, order, gap, reverse);
feats_test.add_preproc(preproc);
feats_test.apply_preproc();

use_sign=false;

kernel=CommWordStringKernel(feats_train, feats_train, use_sign);

km_train=kernel.get_kernel_matrix();
kernel.init(feats_train, feats_test);
km_test=kernel.get_kernel_matrix();

%CommUlongString
disp('CommUlongString')

order=3;
gap=0;
reverse=false;

charfeat=StringCharFeatures(DNA);
charfeat.set_string_features(fm_train_dna);
feats_train=StringUlongFeatures(charfeat.get_alphabet());
feats_train.obtain_from_char(charfeat, order-1, order, gap, reverse);

charfeat=StringCharFeatures(DNA);
charfeat.set_string_features(fm_test_dna);
feats_test=StringUlongFeatures(charfeat.get_alphabet());
feats_test.obtain_from_char(charfeat, order-1, order, gap, reverse);

preproc=SortUlongString();
preproc.init(feats_train);
feats_train.add_preproc(preproc);
feats_train.apply_preproc();
feats_test.add_preproc(preproc);
feats_test.apply_preproc();

use_sign=false;

kernel=CommUlongStringKernel(feats_train, feats_train, use_sign);

km_train=kernel.get_kernel_matrix();
kernel.init(feats_train, feats_test);
km_test=kernel.get_kernel_matrix();
