


degree = 2;
traindat = [rand(100,50)-1 2+rand(100,50)+1];
testdat = [rand(100,50)-1 2+rand(100,50)+1];
trainlab = [ones(1, 50) -ones(1, 50)];


C=1;
size_cache=10; 
epsilon=1e-5;
sg('set_kernel', 'POLY', 'REAL', size_cache, degree);
%sg('set_kernel_normalization', 'IDENTITY');
sg('set_features', 'TRAIN', traindat);
sg('set_labels', 'TRAIN', trainlab);
sg('new_classifier', 'SVMLIGHT');
sg('svm_epsilon', epsilon);
sg('svm_use_bias', 0);
sg('c', C);

sg('init_kernel', 'TRAIN');
km=sg('get_kernel_matrix');
tic; sg('train_classifier'); toc

sg('set_features', 'TEST', testdat);
sg('init_kernel', 'TEST');
result=sg('classify');


normalize=1;
sg('loglevel', 'DEBUG');
sg('svm_use_bias', 0);
sg('set_features', 'TRAIN', traindat, 'POLY', degree, normalize);

x = sg('get_features', 'TRAIN');

km2=x'*x;
keyboard
sg('c', C);
sg('svm_epsilon', epsilon);
sg('svm_use_bias', 0);
sg('new_classifier', 'SVMOCAS');
sg('train_classifier');

sg('set_features', 'TEST', testdat, 'POLY', degree, normalize);
out_wdocas=sg('classify');



