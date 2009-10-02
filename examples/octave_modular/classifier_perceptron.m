init_shogun

addpath('tools');
label_train_twoclass=load_matrix('../data/label_train_twoclass.dat');
fm_train_real=load_matrix('../data/fm_train_real.dat');
fm_test_real=load_matrix('../data/fm_test_real.dat');

% perceptron
disp('Perceptron')

feats_train=RealFeatures(fm_train_real);
feats_test=RealFeatures(fm_train_test);

learn_rate=1.;
max_iter=1000;
num_threads=1;
labels=Labels(label_train_twoclass);

perceptron=Perceptron(feats_train, labels);
perceptron.set_learn_rate(learn_rate);
perceptron.set_max_iter(max_iter);
perceptron.parallel.set_num_threads(num_threads);
perceptron.train();

perceptron.set_features(feats_test);
perceptron.classify().get_labels();

