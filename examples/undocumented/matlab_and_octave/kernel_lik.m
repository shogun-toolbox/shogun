rand('seed',17);
%sequence lengths, number of sequences
len=200;
num_train=500;
num_test=500;
num_a=2;
aa=(round(len/2-num_a/2)):(round(len/2+num_a/2-1));

%SVM regularization factor C
C=1;

%locality improved kernel parameters
cache=100;
l=3;
d1=4;
d2=1;

%generate some toy data
acgt='ACGT';
rand('state',1);
traindat=acgt(ceil(4*rand(len,num_train)));
trainlab=[-ones(1,num_train/2),ones(1,num_train/2)];
traindat(aa,trainlab==1)='A';

testdat=acgt(ceil(4*rand(len,num_test)));
testlab=[-ones(1,num_test/2),ones(1,num_test/2)];
testdat(aa,testlab==1)='A';

%traindat'
%input('key to continue')

%train svm
sg('set_features', 'TRAIN', traindat, 'DNA');
sg('set_labels', 'TRAIN', trainlab);
sg('set_kernel', 'SLIK', 'CHAR', cache, l, d1, d2);
sg('new_classifier', 'LIBSVM');
sg('c', C);
%tic;sg('train_classifier');toc;

%evaluate svm on test data
%sg('set_features', 'TEST', testdat, 'DNA');
%sg('set_labels', 'TEST', testlab);
%out1=sg('classify');
%fprintf('accuracy: %f                                                                                         \n', mean(sign(out1)==testlab))

%out2=sg('classify');
%fprintf('accuracy: %f                                                                                         \n', mean(sign(out2)==testlab))


%tic;out3=sg('classify');toc;
%fprintf('accuracy: %f                                                                                         \n', mean(sign(out3)==testlab))

%max(abs(out1-out2))
%max(abs(out1-out3))
