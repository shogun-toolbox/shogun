% Explicit examples on how to use the different classifiers

num=24;
dist=2.2;
len=42;
size_cache=10;
C=0.017;
use_bias=0;
epsilon=1e-5;
width=2.1;
max_train_time=60;

trainlab_one=[ones(1,num*2) -ones(1,num*2)];
trainlab_multi=[zeros(1,num) ones(1,num) 2*ones(1,num) 3*ones(1,num)];

traindata_real=[randn(2,num)-dist, randn(2,num)+dist, randn(2,num)+dist*[ones(1,num); zeros(1,num)], randn(2,num)+dist*[zeros(1,num); ones(1,num)]];
testdata_real=[randn(2,num+7)-dist, randn(2,num+7)+dist, randn(2,num+7)+dist*[ones(1,num+7); zeros(1,num+7)], randn(2,num+7)+dist*[zeros(1,num+7); ones(1,num+7)]];

acgt='ACGT';
trainlab_dna=[ones(1,num/2) -ones(1,num/2)];
traindata_dna=acgt(ceil(4*rand(len,num)));
testdata_dna=acgt(ceil(4*rand(len,num)));


%
% kernel-based SVMs
%

% SVMLight
disp('SVMLight');

degree=20;
sg('set_features', 'TRAIN', traindata_dna, 'DNA');
sg('send_command', sprintf('set_kernel WEIGHTEDDEGREE CHAR %d %d', size_cache, degree));
sg('send_command', 'init_kernel TRAIN');

sg('set_labels', 'TRAIN', trainlab_dna);
sg('send_command', 'new_svm LIGHT');
sg('send_command', sprintf('svm_epsilon %f', epsilon));
sg('send_command', sprintf('c %f', C));
sg('send_command', sprintf('svm_use_bias %d', use_bias));
sg('send_command', 'svm_train');

sg('set_features', 'TEST', testdata_dna, 'DNA');
sg('send_command', 'init_kernel TEST');
result=sg('svm_classify');


% LibSVM
disp('LibSVM');

sg('set_features', 'TRAIN', traindata_real);
sg('send_command', sprintf('set_kernel GAUSSIAN REAL %d %f', size_cache, width));
sg('send_command', 'init_kernel TRAIN');

sg('set_labels', 'TRAIN', trainlab_one);
sg('send_command', 'new_svm LIBSVM');
sg('send_command', sprintf('svm_epsilon %f', epsilon));
sg('send_command', sprintf('c %f', C));
sg('send_command', sprintf('svm_use_bias %d', use_bias));
sg('send_command', 'svm_train');

sg('set_features', 'TEST', testdata_real);
sg('send_command', 'init_kernel TEST');
result=sg('svm_classify');


% GPBTSVM
disp('GPBTSVM');

sg('set_features', 'TRAIN', traindata_real);
sg('send_command', sprintf('set_kernel GAUSSIAN REAL %d %f', size_cache, width));
sg('send_command', 'init_kernel TRAIN');

sg('set_labels', 'TRAIN', trainlab_one);
sg('send_command', 'new_svm GPBTSVM');
sg('send_command', sprintf('svm_epsilon %f', epsilon));
sg('send_command', sprintf('c %f', C));
sg('send_command', sprintf('svm_use_bias %d', use_bias));
sg('send_command', 'svm_train');

sg('set_features', 'TEST', testdata_real);
sg('send_command', 'init_kernel TEST');
result=sg('svm_classify');


% MPDSVM
disp('MPDSVM');

sg('set_features', 'TRAIN', traindata_real);
sg('send_command', sprintf('set_kernel GAUSSIAN REAL %d %f', size_cache, width));
sg('send_command', 'init_kernel TRAIN');

sg('set_labels', 'TRAIN', trainlab_one);
sg('send_command', 'new_svm MPDSVM');
sg('send_command', sprintf('svm_epsilon %f', epsilon));
sg('send_command', sprintf('c %f', C));
sg('send_command', sprintf('svm_use_bias %d', use_bias));
sg('send_command', 'svm_train');

sg('set_features', 'TEST', testdata_real);
sg('send_command', 'init_kernel TEST');
result=sg('svm_classify');


% LibSVM MultiClass
disp('LibSVMMultiClass');

sg('set_features', 'TRAIN', traindata_real);
sg('send_command', sprintf('set_kernel GAUSSIAN REAL %d %f', size_cache, width));
sg('send_command', 'init_kernel TRAIN');

sg('set_labels', 'TRAIN', trainlab_multi);
sg('send_command', 'new_svm LIBSVM_MULTICLASS');
sg('send_command', sprintf('svm_epsilon %f', epsilon));
sg('send_command', sprintf('c %f', C));
sg('send_command', sprintf('svm_use_bias %d', use_bias));
sg('send_command', 'svm_train');

sg('set_features', 'TEST', testdata_real);
sg('send_command', 'init_kernel TEST');
result=sg('svm_classify');


% LibSVM OneClass
disp('LibSVMOneClass');

sg('set_features', 'TRAIN', traindata_real);
sg('send_command', sprintf('set_kernel GAUSSIAN REAL %d %f', size_cache, width));
sg('send_command', 'init_kernel TRAIN');

sg('send_command', 'new_svm LIBSVM_ONECLASS');
sg('send_command', sprintf('svm_epsilon %f', epsilon));
sg('send_command', sprintf('c %f', C));
sg('send_command', sprintf('svm_use_bias %d', use_bias));
sg('send_command', 'svm_train');

sg('set_features', 'TEST', testdata_real);
sg('send_command', 'init_kernel TEST');
result=sg('svm_classify');


% GMNPSVM
disp('GMNPSVM');

sg('set_features', 'TRAIN', traindata_real);
sg('send_command', sprintf('set_kernel GAUSSIAN REAL %d %f', size_cache, width));
sg('send_command', 'init_kernel TRAIN');

sg('set_labels', 'TRAIN', trainlab_one);
sg('send_command', 'new_svm GMNPSVM');
sg('send_command', sprintf('svm_epsilon %f', epsilon));
sg('send_command', sprintf('c %f', C));
sg('send_command', sprintf('svm_use_bias %d', use_bias));
sg('send_command', 'svm_train');

sg('set_features', 'TEST', testdata_real);
sg('send_command', 'init_kernel TEST');
result=sg('svm_classify');


% run with batch or linadd on LibSVM;
disp('LibSVM batch');

sg('set_features', 'TRAIN', traindata_real);
sg('send_command', sprintf('set_kernel GAUSSIAN REAL %d %f', size_cache, width));
sg('send_command', 'init_kernel TRAIN');

sg('set_labels', 'TRAIN', trainlab_one);
sg('send_command', 'new_svm LIBSVM');
sg('send_command', sprintf('svm_epsilon %f', epsilon));
sg('send_command', sprintf('c %f', C));
sg('send_command', sprintf('svm_use_bias %d', use_bias));
sg('send_command', 'svm_train');

sg('set_features', 'TEST', testdata_real);
sg('send_command', 'init_kernel TEST');

objective=sg('get_svm_objective');
sg('send_command', 'use_batch_computation 1');
sg('send_command', 'use_linadd 1');
result=sg('svm_classify');


%
% SparseLinear classifier
%

% SubgradientSVM - often does not converge
disp('SubGradientSVM');

C=0.42;
sg('set_features', 'TRAIN', sparse(traindata_real));
sg('set_labels', 'TRAIN', trainlab_one);
sg('send_command', 'new_svm SUBGRADIENTSVM');
sg('send_command', sprintf('svm_epsilon %f', epsilon));
sg('send_command', sprintf('c %f', C));
sg('send_command', sprintf('svm_use_bias %d', use_bias));
sg('send_command', sprintf('svm_max_train_time %d', max_train_time));
% sometimes does not terminate
%sg('send_command', 'svm_train');

%sg('set_features', 'TEST', sparse(testdata_real));
%result=sg('svm_classify');

% SVMOcas
disp('SVMOcas');

sg('set_features', 'TRAIN', sparse(traindata_real));
sg('set_labels', 'TRAIN', trainlab_one);
sg('send_command', 'new_svm SVMOCAS');
sg('send_command', sprintf('svm_epsilon %f', epsilon));
sg('send_command', sprintf('c %f', C));
sg('send_command', sprintf('svm_use_bias %d', use_bias));
sg('send_command', 'svm_train');

sg('set_features', 'TEST', sparse(testdata_real));
result=sg('svm_classify');

% SVMSGD
disp('SVMSGD');

sg('set_features', 'TRAIN', sparse(traindata_real));
sg('set_labels', 'TRAIN', trainlab_one);
sg('send_command', 'new_svm SVMSGD');
sg('send_command', sprintf('svm_epsilon %f', epsilon));
sg('send_command', sprintf('c %f', C));
sg('send_command', sprintf('svm_use_bias %d', use_bias));
sg('send_command', sprintf('svm_max_train_time %d', max_train_time));
sg('send_command', 'svm_train');

sg('set_features', 'TEST', sparse(testdata_real));
result=sg('svm_classify');

% LibLinear
disp('LibLinear');

sg('set_features', 'TRAIN', sparse(traindata_real));
sg('set_labels', 'TRAIN', trainlab_one);
sg('send_command', 'new_svm LIBLINEAR_LR');
sg('send_command', sprintf('svm_epsilon %f', epsilon));
sg('send_command', sprintf('c %f', C));
sg('send_command', sprintf('svm_use_bias %d', use_bias));
sg('send_command', sprintf('svm_max_train_time %d', max_train_time));
sg('send_command', 'svm_train');

sg('set_features', 'TEST', sparse(testdata_real));
result=sg('svm_classify');

% SVMLin
disp('SVMLin');

sg('set_features', 'TRAIN', sparse(traindata_real));
sg('set_labels', 'TRAIN', trainlab_one);
sg('send_command', 'new_svm SVMLIN');
sg('send_command', sprintf('svm_epsilon %f', epsilon));
sg('send_command', sprintf('c %f', C));
sg('send_command', sprintf('svm_use_bias %d', use_bias));
sg('send_command', sprintf('svm_max_train_time %d', max_train_time));
sg('send_command', 'svm_train');

sg('set_features', 'TEST', sparse(testdata_real));
result=sg('svm_classify');


%
% misc classifiers
%

% Perceptron
disp('Perceptron');

sg('set_features', 'TRAIN', traindata_real);
sg('set_labels', 'TRAIN', trainlab_one);
sg('send_command', 'new_classifier PERCEPTRON');
sg('send_command', 'train_classifier');

sg('set_features', 'TEST', testdata_real);
result=sg('classify');

% KNN
disp('KNN');

sg('set_features', 'TRAIN', traindata_real);
sg('set_labels', 'TRAIN', trainlab_one);
sg('send_command', 'set_distance EUCLIDIAN REAL');
sg('send_command', 'init_distance TRAIN');
sg('send_command', 'new_knn');
sg('send_command', 'train_knn');

sg('set_features', 'TEST', testdata_real);
sg('send_command', 'init_distance TEST');
result=sg('classify');


% LDA
disp('LDA');

sg('set_features', 'TRAIN', traindata_real);
sg('set_labels', 'TRAIN', trainlab_one);
sg('send_command', 'new_classifier LDA');
sg('send_command', 'train_classifier');

sg('set_features', 'TEST', testdata_real);
result=sg('classify');

