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
try
	sg('new_svm', 'LIGHT');

	disp('SVMLight');

	degree=20;
	sg('set_features', 'TRAIN', traindata_dna, 'DNA');
	sg('set_kernel', 'WEIGHTEDDEGREE', 'CHAR', size_cache, degree);
	sg('init_kernel', 'TRAIN');

	sg('set_labels', 'TRAIN', trainlab_dna);

	sg('svm_epsilon', epsilon);
	sg('c', C);
	sg('svm_use_bias', use_bias);
	sg('train_classifier');

	sg('set_features', 'TEST', testdata_dna, 'DNA');
	sg('init_kernel', 'TEST');
	result=sg('classify');
catch
	disp('No support for SVMLight available.')
end


% LibSVM
disp('LibSVM');

sg('set_features', 'TRAIN', traindata_real);
sg('set_kernel', 'GAUSSIAN', 'REAL', size_cache, width);
sg('init_kernel', 'TRAIN');

sg('set_labels', 'TRAIN', trainlab_one);
sg('new_svm', 'LIBSVM');
sg('svm_epsilon', epsilon);
sg('c', C);
sg('svm_use_bias', use_bias);
sg('train_classifier');

sg('set_features', 'TEST', testdata_real);
sg('init_kernel', 'TEST');
result=sg('classify');


% GPBTSVM
disp('GPBTSVM');

sg('set_features', 'TRAIN', traindata_real);
sg('set_kernel', 'GAUSSIAN', 'REAL', size_cache, width);
sg('init_kernel', 'TRAIN');

sg('set_labels', 'TRAIN', trainlab_one);
sg('new_svm', 'GPBTSVM');
sg('svm_epsilon', epsilon);
sg('c', C);
sg('svm_use_bias', use_bias);
sg('train_classifier');

sg('set_features', 'TEST', testdata_real);
sg('init_kernel', 'TEST');
result=sg('classify');

% MPDSVM
disp('MPDSVM');

sg('set_features', 'TRAIN', traindata_real);
sg('set_kernel', 'GAUSSIAN', 'REAL', size_cache, width);
sg('init_kernel', 'TRAIN');

sg('set_labels', 'TRAIN', trainlab_one);
sg('new_svm', 'MPDSVM');
sg('svm_epsilon', epsilon);
sg('c', C);
sg('svm_use_bias', use_bias);
sg('train_classifier');

sg('set_features', 'TEST', testdata_real);
sg('init_kernel', 'TEST');
result=sg('classify');


% LibSVM MultiClass
disp('LibSVMMultiClass');

sg('set_features', 'TRAIN', traindata_real);
sg('set_kernel', 'GAUSSIAN', 'REAL', size_cache, width);
sg('init_kernel', 'TRAIN');

sg('set_labels', 'TRAIN', trainlab_multi);
sg('new_svm', 'LIBSVM_MULTICLASS');
sg('svm_epsilon', epsilon);
sg('c', C);
sg('svm_use_bias', use_bias);
sg('train_classifier');

sg('set_features', 'TEST', testdata_real);
sg('init_kernel', 'TEST');
result=sg('classify');


% LibSVM OneClass
disp('LibSVMOneClass');

sg('set_features', 'TRAIN', traindata_real);
sg('set_kernel', 'GAUSSIAN', 'REAL', size_cache, width);
sg('init_kernel', 'TRAIN');

sg('new_svm', 'LIBSVM_ONECLASS');
sg('svm_epsilon', epsilon);
sg('c', C);
sg('svm_use_bias', use_bias);
sg('train_classifier');

sg('set_features', 'TEST', testdata_real);
sg('init_kernel', 'TEST');
result=sg('classify');


% GMNPSVM
disp('GMNPSVM');

sg('set_features', 'TRAIN', traindata_real);
sg('set_kernel', 'GAUSSIAN', 'REAL', size_cache, width);
sg('init_kernel', 'TRAIN');

sg('set_labels', 'TRAIN', trainlab_one);
sg('new_svm', 'GMNPSVM');
sg('svm_epsilon', epsilon);
sg('c', C);
sg('svm_use_bias', use_bias);
sg('train_classifier');

sg('set_features', 'TEST', testdata_real);
sg('init_kernel', 'TEST');
result=sg('classify');


% run with batch or linadd on LibSVM;
disp('LibSVM batch');

sg('set_features', 'TRAIN', traindata_real);
sg('set_kernel', 'GAUSSIAN', 'REAL', size_cache, width);
sg('init_kernel', 'TRAIN');

sg('set_labels', 'TRAIN', trainlab_one);
sg('new_svm', 'LIBSVM');
sg('svm_epsilon', epsilon);
sg('c', C);
sg('svm_use_bias', use_bias);
sg('train_classifier');

sg('set_features', 'TEST', testdata_real);
sg('init_kernel', 'TEST');
result=sg('classify');

objective=sg('get_svm_objective');
sg('use_batch_computation', 1);
sg('use_linadd', 1);
result=sg('classify');


%
% SparseLinear classifier
%

% SubgradientSVM - often does not converge
disp('SubGradientSVM');

C=0.42;
sg('set_features', 'TRAIN', sparse(traindata_real));
sg('set_labels', 'TRAIN', trainlab_one);
sg('new_svm', 'SUBGRADIENTSVM');
sg('svm_epsilon', epsilon);
sg('c', C);
sg('svm_use_bias', use_bias);
sg('svm_max_train_time', max_train_time);
% sometimes does not terminate
%sg('train_classifier');

%sg('set_features', 'TEST', sparse(testdata_real));
%result=sg('classify');

% SVMOcas
disp('SVMOcas');

sg('set_features', 'TRAIN', sparse(traindata_real));
sg('set_labels', 'TRAIN', trainlab_one);
sg('new_svm', 'SVMOCAS');
sg('svm_epsilon', epsilon);
sg('c', C);
sg('svm_use_bias', use_bias);
sg('svm_max_train_time', max_train_time);
sg('train_classifier');

sg('set_features', 'TEST', sparse(testdata_real));
result=sg('classify');

% SVMSGD
disp('SVMSGD');

sg('set_features', 'TRAIN', sparse(traindata_real));
sg('set_labels', 'TRAIN', trainlab_one);
sg('new_svm', 'SVMSGD');
sg('svm_epsilon', epsilon);
sg('c', C);
sg('svm_use_bias', use_bias);
sg('svm_max_train_time', max_train_time);
sg('train_classifier');

sg('set_features', 'TEST', sparse(testdata_real));
result=sg('classify');

% LibLinear
disp('LibLinear');

sg('set_features', 'TRAIN', sparse(traindata_real));
sg('set_labels', 'TRAIN', trainlab_one);
sg('new_svm', 'LIBLINEAR_LR');
sg('svm_epsilon', epsilon);
sg('c', C);
sg('svm_use_bias', use_bias);
sg('svm_max_train_time', max_train_time);
sg('train_classifier');

sg('set_features', 'TEST', sparse(testdata_real));
result=sg('classify');

% SVMLin
disp('SVMLin');

sg('set_features', 'TRAIN', sparse(traindata_real));
sg('set_labels', 'TRAIN', trainlab_one);
sg('new_svm', 'SVMLIN');
sg('svm_epsilon', epsilon);
sg('c', C);
sg('svm_use_bias', use_bias);
sg('svm_max_train_time', max_train_time);
sg('train_classifier');

sg('set_features', 'TEST', sparse(testdata_real));
result=sg('classify');


%
% misc classifiers
%

% Perceptron
disp('Perceptron');

trainlab_one=[ones(1,num) -ones(1,num)];
traindata_real=[randn(2,num)-dist, randn(2,num)+dist];
sg('set_features', 'TRAIN', traindata_real);
sg('set_labels', 'TRAIN', trainlab_one);
sg('new_classifier', 'PERCEPTRON');
sg('train_classifier');

sg('set_features', 'TEST', testdata_real);
result=sg('classify');

% KNN
disp('KNN');

sg('set_features', 'TRAIN', traindata_real);
sg('set_labels', 'TRAIN', trainlab_one);
sg('set_distance', 'EUCLIDIAN', 'REAL');
sg('init_distance', 'TRAIN');
sg('new_classifier', 'KNN');
sg('train_classifier', 3);

sg('set_features', 'TEST', testdata_real);
sg('init_distance', 'TEST');
result=sg('classify');


% LDA
disp('LDA');

sg('set_features', 'TRAIN', traindata_real);
sg('set_labels', 'TRAIN', trainlab_one);
sg('new_classifier', 'LDA');
sg('train_classifier');

sg('set_features', 'TEST', testdata_real);
result=sg('classify');

