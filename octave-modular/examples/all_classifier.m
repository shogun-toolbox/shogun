init_shogun

num=40;
len=3;
dist=2;

% Explicit examples on how to use the different classifiers

acgt='ACGT';
trainlab_dna=[ones(1,num/2) -ones(1,num/2)];
traindata_dna=acgt(ceil(4*rand(len,num)));
testdata_dna=acgt(ceil(4*rand(len,num)));
trainlab_multi=[zeros(1,num/4) ones(1,num/4) 2*ones(1,num/4) 3*ones(1,num/4)];

traindata_real=[randn(2,num)-dist, randn(2,num)+dist];
testdata_real=[randn(2,num+7)-dist, randn(2,num+7)+dist];

maxval=2^16-1;
traindata_word=uint16(rand(len, num)*maxval);
testdata_word=uint16(rand(len, num)*maxval);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% kernel-based SVMs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% svm light
disp('SVMLight')

feats_train=StringCharFeatures(DNA);
feats_train.set_string_features(traindata_dna);
feats_test=StringCharFeatures(DNA);
feats_test.set_string_features(testdata_dna);
degree=20;

kernel=WeightedDegreeStringKernel(feats_train, feats_train, degree);

C=0.017;
epsilon=1e-5;
tube_epsilon=1e-2;
num_threads=3;
lab=round(rand(1,feats_train.get_num_vectors()))*2-1;
labels=Labels(lab);

svm=SVMLight(C, kernel, labels);
svm.set_epsilon(epsilon);
svm.set_tube_epsilon(tube_epsilon);
svm.parallel.set_num_threads(num_threads);
svm.train();

kernel.init(feats_train, feats_test);
svm.classify().get_labels();

% libsvm
disp('LibSVM')

feats_train=RealFeatures(traindata_real);
feats_test=RealFeatures(testdata_real);
width=2.1;
kernel=GaussianKernel(feats_train, feats_train, width);

C=0.017;
epsilon=1e-5;
tube_epsilon=1e-2;
num_threads=2;
lab=round(rand(1,feats_train.get_num_vectors()))*2-1;
labels=Labels(lab);

svm=LibSVM(C, kernel, labels);
svm.set_epsilon(epsilon);
svm.set_tube_epsilon(tube_epsilon);
svm.parallel.set_num_threads(num_threads);
svm.train();

kernel.init(feats_train, feats_test);
svm.classify().get_labels();

% gpbtsvm
disp('GPBTSVM')

feats_train=RealFeatures(traindata_real);
feats_test=RealFeatures(testdata_real);
width=2.1;
kernel=GaussianKernel(feats_train, feats_train, width);

C=0.017;
epsilon=1e-5;
tube_epsilon=1e-2;
num_threads=2;
lab=round(rand(1,feats_train.get_num_vectors()))*2-1;
labels=Labels(lab);

svm=GPBTSVM(C, kernel, labels);
svm.set_epsilon(epsilon);
svm.set_tube_epsilon(tube_epsilon);
svm.parallel.set_num_threads(num_threads);
svm.train();

kernel.init(feats_train, feats_test);
svm.classify().get_labels();

% mpdsvm
disp('MPDSVM')

feats_train=RealFeatures(traindata_real);
feats_test=RealFeatures(testdata_real);
width=2.1;
kernel=GaussianKernel(feats_train, feats_train, width);

C=0.017;
epsilon=1e-5;
tube_epsilon=1e-2;
num_threads=1
lab=round(rand(1,feats_train.get_num_vectors()))*2-1;
labels=Labels(lab);

svm=MPDSVM(C, kernel, labels);
svm.set_epsilon(epsilon);
svm.set_tube_epsilon(tube_epsilon);
svm.parallel.set_num_threads(num_threads);
svm.train();

kernel.init(feats_train, feats_test);
svm.classify().get_labels();

% libsvmmulticlass
disp('LibSVMMultiClass')

feats_train=RealFeatures(traindata_real);
feats_test=RealFeatures(testdata_real);
width=2.1;
kernel=GaussianKernel(feats_train, feats_train, width);

C=0.017;
epsilon=1e-5;
tube_epsilon=1e-2;
num_threads=8;
labels=Labels(trainlab_multi);

svm=LibSVMMultiClass(C, kernel, labels);
svm.set_epsilon(epsilon);
svm.set_tube_epsilon(tube_epsilon);
svm.parallel.set_num_threads(num_threads);
svm.train();

kernel.init(feats_train, feats_test);
svm.classify().get_labels();

% libsvm oneclass
disp('LibSVMOneClass')

feats_train=RealFeatures(traindata_real);
feats_test=RealFeatures(testdata_real);
width=2.1;
kernel=GaussianKernel(feats_train, feats_train, width);

C=0.017;
epsilon=1e-5;
tube_epsilon=1e-2;
num_threads=4;

svm=LibSVMOneClass(C, kernel);
svm.set_epsilon(epsilon);
svm.set_tube_epsilon(tube_epsilon);
svm.parallel.set_num_threads(num_threads);
svm.train();

kernel.init(feats_train, feats_test);
svm.classify().get_labels();

% gmnpsvm
disp('GMNPSVM')

feats_train=RealFeatures(traindata_real);
feats_test=RealFeatures(testdata_real);
width=2.1;
kernel=GaussianKernel(feats_train, feats_train, width);

C=0.017;
epsilon=1e-5;
tube_epsilon=1e-2;
num_threads=1;
labels=Labels(trainlab_multi);

svm=GMNPSVM(C, kernel, labels);
svm.set_epsilon(epsilon);
svm.set_tube_epsilon(tube_epsilon);
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
feats_train.set_string_features(traindata_dna);
feats_test=StringCharFeatures(DNA);
feats_test.set_string_features(testdata_dna);
degree=20;

kernel=WeightedDegreeStringKernel(feats_train, feats_train, degree);

C=0.017;
epsilon=1e-5;
tube_epsilon=1e-2;
num_threads=2;
lab=round(rand(1,feats_train.get_num_vectors()))*2-1;
labels=Labels(lab);

svm=LibSVM(C, kernel, labels);
svm.set_epsilon(epsilon);
svm.set_tube_epsilon(tube_epsilon);
svm.parallel.set_num_threads(num_threads);
svm.train();

kernel.init(feats_train, feats_test);

fprintf('LibSVM Objective: %f num_sv: %d', svm.get_objective(), svm.get_num_support_vectors())
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

realfeat=RealFeatures(traindata_real);
feats_train=SparseRealFeatures();
feats_train.obtain_from_simple(realfeat);
realfeat=RealFeatures(testdata_real);
feats_test=SparseRealFeatures();
feats_test.obtain_from_simple(realfeat);

C=0.42;
epsilon=1e-3;
num_threads=1;
max_train_time=1.;
lab=round(rand(1, feats_train.get_num_vectors()))*2-1;
labels=Labels(lab);

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

realfeat=RealFeatures(traindata_real);
feats_train=SparseRealFeatures();
feats_train.obtain_from_simple(realfeat);
realfeat=RealFeatures(testdata_real);
feats_test=SparseRealFeatures();
feats_test.obtain_from_simple(realfeat);

C=0.42;
epsilon=1e-5;
num_threads=1;
lab=round(rand(1, feats_train.get_num_vectors()))*2-1;
labels=Labels(lab);

svm=SVMOcas(C, feats_train, labels);
svm.set_epsilon(epsilon);
svm.parallel.set_num_threads(num_threads);
svm.set_bias_enabled(false);
svm.train();

svm.set_features(feats_test);
svm.classify().get_labels();

% sgd
disp('SVMSGD')

realfeat=RealFeatures(traindata_real);
feats_train=SparseRealFeatures();
feats_train.obtain_from_simple(realfeat);
realfeat=RealFeatures(testdata_real);
feats_test=SparseRealFeatures();
feats_test.obtain_from_simple(realfeat);

C=0.42;
epsilon=1e-5;
num_threads=1;
lab=round(rand(1,feats_test.get_num_vectors()))*2-1;
labels=Labels(lab);

svm=SVMSGD(C, feats_test, labels);
svm.io.set_loglevel(0);
svm.train();

svm.set_features(feats_test);
svm.classify().get_labels();

% liblinear
disp('LibLinear')

realfeat=RealFeatures(traindata_real);
feats_train=SparseRealFeatures();
feats_train.obtain_from_simple(realfeat);
realfeat=RealFeatures(testdata_real);
feats_test=SparseRealFeatures();
feats_test.obtain_from_simple(realfeat);

C=0.42;
epsilon=1e-5;
num_threads=1;
lab=round(rand(1,feats_train.get_num_vectors()))*2-1;
labels=Labels(lab);

svm=LibLinear(C, feats_train, labels);
svm.set_epsilon(epsilon);
svm.parallel.set_num_threads(num_threads);
svm.set_bias_enabled(true);
svm.train();

svm.set_features(feats_test);
svm.classify().get_labels();

% svm lin
disp('SVMLin')

realfeat=RealFeatures(traindata_real);
feats_train=SparseRealFeatures();
feats_train.obtain_from_simple(realfeat);
realfeat=RealFeatures(testdata_real);
feats_test=SparseRealFeatures();
feats_test.obtain_from_simple(realfeat);

C=0.42;
epsilon=1e-5;
num_threads=1;
lab=round(rand(1, feats_train.get_num_vectors()))*2-1;
labels=Labels(lab);

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

feats_train=RealFeatures(traindata_real);
feats_test=RealFeatures(traindata_real);

learn_rate=1.;
max_iter=1000;
num_threads=1;
lab=round(rand(1, feats_train.get_num_vectors()))*2-1;
labels=Labels(lab);

perceptron=Perceptron(feats_train, labels);
perceptron.set_learn_rate(learn_rate);
perceptron.set_max_iter(max_iter);
perceptron.parallel.set_num_threads(num_threads);
perceptron.train();

perceptron.set_features(feats_test);
perceptron.classify().get_labels();

% knn
disp('KNN')

feats_train=RealFeatures(traindata_real);
feats_test=RealFeatures(testdata_real);
distance=EuclidianDistance();

k=3;
num_threads=1;
lab=round(rand(1, feats_train.get_num_vectors()))*2-1;
labels=Labels(lab);

knn=KNN(k, distance, labels);
knn.parallel.set_num_threads(num_threads);
knn.train();

distance.init(feats_train, feats_test);
knn.classify().get_labels();

% lda
disp('LDA')

feats_train=RealFeatures(traindata_real);
feats_test=RealFeatures(testdata_real);

gamma=3;
num_threads=1;
lab=round(rand(1, feats_train.get_num_vectors()))*2-1;
labels=Labels(lab);

lda=LDA(gamma, feats_train, labels);
lda.parallel.set_num_threads(num_threads);
lda.train();

lda.get_bias();
lda.get_w();
lda.set_features(feats_test);
lda.classify().get_labels();
