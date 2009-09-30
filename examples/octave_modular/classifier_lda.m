init_shogun

addpath('tools');
label_train_twoclass=load_matrix('../data/label_train_twoclass.dat');
label_train_multiclass=load_matrix('../data/label_train_multiclass.dat');
fm_train_real=load_matrix('../data/fm_train_real.dat');
fm_test_real=load_matrix('../data/fm_test_real.dat');

label_train_dna=load_matrix('../data/label_train_dna.dat');
fm_train_dna=load_matrix('../data/fm_train_dna.dat');
fm_test_dna=load_matrix('../data/fm_test_dna.dat');

% lda
disp('LDA')

feats_train=RealFeatures(fm_train_real);
feats_test=RealFeatures(fm_test_real);

gamma=3;
num_threads=1;
labels=Labels(label_train_twoclass);

lda=LDA(gamma, feats_train, labels);
lda.parallel.set_num_threads(num_threads);
lda.train();

lda.get_bias();
lda.get_w();
lda.set_features(feats_test);
lda.classify().get_labels();
