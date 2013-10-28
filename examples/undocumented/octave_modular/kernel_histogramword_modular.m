modshogun

addpath('tools');
label_train_dna=load_matrix('../data/label_train_dna.dat');
fm_train_dna=load_matrix('../data/fm_train_dna.dat');
fm_test_dna=load_matrix('../data/fm_test_dna.dat');

% plugin_estimate
disp('PluginEstimate w/ HistogramWord')

order=3;
gap=0;
reverse=false;

charfeat=StringCharFeatures(DNA);
charfeat.set_features(fm_train_dna);
feats_train=StringWordFeatures(charfeat.get_alphabet());
feats_train.obtain_from_char(charfeat, order-1, order, gap, reverse);

charfeat=StringCharFeatures(DNA);
charfeat.set_features(fm_test_dna);
feats_test=StringWordFeatures(charfeat.get_alphabet());
feats_test.obtain_from_char(charfeat, order-1, order, gap, reverse);

pie=PluginEstimate();
labels=BinaryLabels(label_train_dna);
pie.set_labels(labels);
pie.set_features(feats_train);
pie.train();

kernel=HistogramWordStringKernel(feats_train, feats_train, pie);
km_train=kernel.get_kernel_matrix();

kernel.init(feats_train, feats_test);
pie.set_features(feats_test);
pie.apply().get_labels();
km_test=kernel.get_kernel_matrix();

