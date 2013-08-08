modshogun

addpath('tools');
fm_train_dna=load_matrix('../data/fm_train_dna.dat');
fm_test_dna=load_matrix('../data/fm_test_dna.dat');

order=4;
gap=0;
reverse=false;
use_sign=true;

% comm_word_string
disp('CommWordString')

charfeat=StringCharFeatures(DNA);
charfeat.set_features(fm_train_dna);
feats_train=StringWordFeatures(charfeat.get_alphabet());
feats_train.obtain_from_char(charfeat, order-1, order, gap, reverse);
preproc=SortWordString();
preproc.init(feats_train);
feats_train.add_preprocessor(preproc);
feats_train.apply_preprocessor();

charfeat=StringCharFeatures(DNA);
charfeat.set_features(fm_test_dna);
feats_test=StringWordFeatures(charfeat.get_alphabet());
feats_test.obtain_from_char(charfeat, order-1, order, gap, reverse);
feats_test.add_preprocessor(preproc);
feats_test.apply_preprocessor();

use_sign=false;

kernel=CommWordStringKernel(feats_train, feats_train, use_sign);

km_train=kernel.get_kernel_matrix();
kernel.init(feats_train, feats_test);
km_test=kernel.get_kernel_matrix();

