init_shogun

addpath('tools');
fm_train_dna=load_matrix('../data/fm_train_dna.dat');
fm_test_dna=load_matrix('../data/fm_test_dna.dat');

%CommUlongString
disp('CommUlongString')

order=3;
gap=0;
reverse=false;

charfeat=StringCharFeatures(DNA);
charfeat.set_features(fm_train_dna);
feats_train=StringUlongFeatures(charfeat.get_alphabet());
feats_train.obtain_from_char(charfeat, order-1, order, gap, reverse);

charfeat=StringCharFeatures(DNA);
charfeat.set_features(fm_test_dna);
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
