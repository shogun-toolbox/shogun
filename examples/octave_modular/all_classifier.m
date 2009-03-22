init_shogun

% Explicit examples on how to use the different classifiers

addpath('tools');
label_train_twoclass=load_matrix('../data/label_train_twoclass.dat');
label_train_multiclass=load_matrix('../data/label_train_multiclass.dat');
fm_train_real=load_matrix('../data/fm_train_real.dat');
fm_test_real=load_matrix('../data/fm_test_real.dat');

label_train_dna=load_matrix('../data/label_train_dna.dat');
fm_train_dna=load_matrix('../data/fm_train_dna.dat');
fm_test_dna=load_matrix('../data/fm_test_dna.dat');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% kernel-based SVMs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% svm light
if exist('SVMLight')
	disp('SVMLight')

	feats_train=StringCharFeatures(DNA);
	feats_train.set_features(fm_train_dna);
	feats_test=StringCharFeatures(DNA);
	feats_test.set_features(fm_test_dna);
	degree=20;

	kernel=WeightedDegreeStringKernel(feats_train, feats_train, degree);

	C=0.017;
	epsilon=1e-5;
	num_threads=3;
	labels=Labels(label_train_dna);

	svm=SVMLight(C, kernel, labels);
	svm.set_epsilon(epsilon);
	svm.parallel.set_num_threads(num_threads);
	svm.train();

	kernel.init(feats_train, feats_test);
	svm.classify().get_labels();
else
	disp('No support for SVMLight available.')
end


% libsvm
disp('LibSVM')

feats_train=RealFeatures(fm_train_real);
feats_test=RealFeatures(fm_test_real);
width=2.1;
kernel=GaussianKernel(feats_train, feats_train, width);

C=0.017;
epsilon=1e-5;
num_threads=2;
labels=Labels(label_train_twoclass);

svm=LibSVM(C, kernel, labels);
svm.set_epsilon(epsilon);
svm.parallel.set_num_threads(num_threads);
svm.train();

kernel.init(feats_train, feats_test);
svm.classify().get_labels();

% gpbtsvm
disp('GPBTSVM')

feats_train=RealFeatures(fm_train_real);
feats_test=RealFeatures(fm_test_real);
width=2.1;
kernel=GaussianKernel(feats_train, feats_train, width);

C=0.017;
epsilon=1e-5;
num_threads=2;
labels=Labels(label_train_twoclass);

svm=GPBTSVM(C, kernel, labels);
svm.set_epsilon(epsilon);
svm.parallel.set_num_threads(num_threads);
svm.train();

kernel.init(feats_train, feats_test);
svm.classify().get_labels();

% mpdsvm
disp('MPDSVM')

feats_train=RealFeatures(fm_train_real);
feats_test=RealFeatures(fm_test_real);
width=2.1;
kernel=GaussianKernel(feats_train, feats_train, width);

C=0.017;
epsilon=1e-5;
num_threads=1;
labels=Labels(label_train_twoclass);

svm=MPDSVM(C, kernel, labels);
svm.set_epsilon(epsilon);
svm.parallel.set_num_threads(num_threads);
svm.train();

kernel.init(feats_train, feats_test);
svm.classify().get_labels();

% libsvmmulticlass
disp('LibSVMMultiClass')

feats_train=RealFeatures(fm_train_real);
feats_test=RealFeatures(fm_test_real);
width=2.1;
kernel=GaussianKernel(feats_train, feats_train, width);

C=0.017;
epsilon=1e-5;
num_threads=8;
labels=Labels(label_train_multiclass);

svm=LibSVMMultiClass(C, kernel, labels);
svm.set_epsilon(epsilon);
svm.parallel.set_num_threads(num_threads);
svm.train();

kernel.init(feats_train, feats_test);
svm.classify().get_labels();

% libsvm twoclass
disp('LibSVMOneClass')

feats_train=RealFeatures(fm_train_real);
feats_test=RealFeatures(fm_test_real);
width=2.1;
kernel=GaussianKernel(feats_train, feats_train, width);

C=0.017;
epsilon=1e-5;
num_threads=4;

svm=LibSVMOneClass(C, kernel);
svm.set_epsilon(epsilon);
svm.parallel.set_num_threads(num_threads);
svm.train();

kernel.init(feats_train, feats_test);
svm.classify().get_labels();

% gmnpsvm
disp('GMNPSVM')

feats_train=RealFeatures(fm_train_real);
feats_test=RealFeatures(fm_test_real);
width=2.1;
kernel=GaussianKernel(feats_train, feats_train, width);

C=0.017;
epsilon=1e-5;
num_threads=1;
labels=Labels(label_train_multiclass);

svm=GMNPSVM(C, kernel, labels);
svm.set_epsilon(epsilon);
svm.parallel.set_num_threads(num_threads);
svm.train();

kernel.init(feats_train, feats_test);
svm.classify().get_labels();

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% run with batch or linadd on LibSVM
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% batch & linadd
disp('LibSVM batch')

feats_train=StringCharFeatures(DNA);
feats_train.set_features(fm_train_dna);
feats_test=StringCharFeatures(DNA);
feats_test.set_features(fm_test_dna);
degree=20;

kernel=WeightedDegreeStringKernel(feats_train, feats_train, degree);

C=0.017;
epsilon=1e-5;
num_threads=2;
labels=Labels(label_train_dna);

svm=LibSVM(C, kernel, labels);
svm.set_epsilon(epsilon);
svm.parallel.set_num_threads(num_threads);
svm.train();

kernel.init(feats_train, feats_test);

%fprintf('LibSVM Objective: %f num_sv: %d', svm.get_objective(), svm.get_num_support_vectors())
svm.set_batch_computation_enabled(false);
svm.set_linadd_enabled(false);
svm.classify().get_labels();

svm.set_batch_computation_enabled(true);
svm.classify().get_labels();

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% linear SVMs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% subgradient based svm
disp('SubGradientSVM')

realfeat=RealFeatures(fm_train_real);
feats_train=SparseRealFeatures();
feats_train.obtain_from_simple(realfeat);
realfeat=RealFeatures(fm_test_real);
feats_test=SparseRealFeatures();
feats_test.obtain_from_simple(realfeat);

C=0.42;
epsilon=1e-3;
num_threads=1;
max_train_time=1.;
labels=Labels(label_train_twoclass);

svm=SubGradientSVM(C, feats_train, labels);
svm.set_epsilon(epsilon);
svm.parallel.set_num_threads(num_threads);
svm.set_bias_enabled(false);
svm.set_max_train_time(max_train_time);
svm.train();

svm.set_features(feats_test);
svm.classify().get_labels();

% svm ocas
disp('SVMOcas')

realfeat=RealFeatures(fm_train_real);
feats_train=SparseRealFeatures();
feats_train.obtain_from_simple(realfeat);
realfeat=RealFeatures(fm_test_real);
feats_test=SparseRealFeatures();
feats_test.obtain_from_simple(realfeat);

C=0.42;
epsilon=1e-5;
num_threads=1;
labels=Labels(label_train_twoclass);

svm=SVMOcas(C, feats_train, labels);
svm.set_epsilon(epsilon);
svm.parallel.set_num_threads(num_threads);
svm.set_bias_enabled(false);
svm.train();

svm.set_features(feats_test);
svm.classify().get_labels();

% sgd
disp('SVMSGD')

realfeat=RealFeatures(fm_train_real);
feats_train=SparseRealFeatures();
feats_train.obtain_from_simple(realfeat);
realfeat=RealFeatures(fm_test_real);
feats_test=SparseRealFeatures();
feats_test.obtain_from_simple(realfeat);

C=0.42;
epsilon=1e-5;
num_threads=1;
labels=Labels(label_train_twoclass);

svm=SVMSGD(C, feats_train, labels);
%svm.io.set_loglevel(0);
svm.train();

svm.set_features(feats_test);
svm.classify().get_labels();

% liblinear
disp('LibLinear')

realfeat=RealFeatures(fm_train_real);
feats_train=SparseRealFeatures();
feats_train.obtain_from_simple(realfeat);
realfeat=RealFeatures(fm_test_real);
feats_test=SparseRealFeatures();
feats_test.obtain_from_simple(realfeat);

C=0.42;
epsilon=1e-5;
num_threads=1;
labels=Labels(label_train_twoclass);

svm=LibLinear(C, feats_train, labels);
svm.set_epsilon(epsilon);
svm.parallel.set_num_threads(num_threads);
svm.set_bias_enabled(true);
svm.train();

svm.set_features(feats_test);
svm.classify().get_labels();

% svm lin
disp('SVMLin')

realfeat=RealFeatures(fm_train_real);
feats_train=SparseRealFeatures();
feats_train.obtain_from_simple(realfeat);
realfeat=RealFeatures(fm_test_real);
feats_test=SparseRealFeatures();
feats_test.obtain_from_simple(realfeat);

C=0.42;
epsilon=1e-5;
num_threads=1;
labels=Labels(label_train_twoclass);

svm=SVMLin(C, feats_train, labels);
svm.set_epsilon(epsilon);
svm.parallel.set_num_threads(num_threads);
svm.set_bias_enabled(true);
svm.train();

svm.set_features(feats_test);
svm.get_bias();
svm.get_w();
svm.classify().get_labels();

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% misc classifiers
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% perceptron
disp('Perceptron')

feats_train=RealFeatures(fm_train_real);
feats_test=RealFeatures(fm_train_real);

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

% knn
disp('KNN')

feats_train=RealFeatures(fm_train_real);
feats_test=RealFeatures(fm_test_real);
distance=EuclidianDistance();

k=3;
num_threads=1;
labels=Labels(label_train_twoclass);

knn=KNN(k, distance, labels);
knn.parallel.set_num_threads(num_threads);
knn.train();

distance.init(feats_train, feats_test);
knn.classify().get_labels();

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
