C=1;
order=20;
order_com=5;
max_mismatch=0;
len=200;
shift=10;
num=1000;
num_test=2000;
cache=10;

acgt='ACGT';
rand('state',1);
traindat=acgt(ceil(4*rand(len,num)));
trainlab=[-ones(1,num/2),ones(1,num/2)];

testdat=acgt(ceil(4*rand(len,num_test)));
testlab=[-ones(1,num/2),ones(1,num_test/2)];
x=ceil(linspace(1,shift,len));
shifts = int32(x(end:-1:1));

%sg('loglevel', 'ALL');
sg('clean_features', 'TRAIN');
sg('clean_features', 'TEST');
sg('threads', 4);
sg('clean_kernel');
sg('use_linadd', 1);
sg('use_batch_computation', 1);

sg('set_features', 'TRAIN', traindat, 'DNA');
sg('set_labels', 'TRAIN', trainlab);

sg('set_features', 'TEST', testdat, 'DNA');
sg('set_labels', 'TEST', testlab);
%
sg('set_kernel', 'WEIGHTEDDEGREEPOS2', 'CHAR', 10', order, max_mismatch, len, shifts);
sg('set_kernel_optimization_type', 'FASTBUTMEMHUNGRY');
kt=sg('get_kernel_matrix', 'TEST');
sg('new_classifier', 'SVMLIGHT');
sg('c', C);
tic; sg('train_classifier'); t=toc
[b, alphas]=sg('get_svm');

tic;
outopt=sg('classify');
tout=toc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
sg('loglevel', 'ALL');
sg('clean_features', 'TRAIN');
sg('clean_features', 'TEST');
sg('clean_kernel');
sg('threads', 4);
sg('use_linadd', 0);
sg('use_batch_computation', 0);

sg('set_features', 'TRAIN', traindat, 'DNA');
sg('set_labels', 'TRAIN', trainlab);
%
sg('set_features', 'TEST', testdat, 'DNA');
sg('set_labels', 'TEST', testlab);
%
sg('set_kernel', 'WEIGHTEDDEGREEPOS2', 'CHAR', 10, order, max_mismatch, len, shifts);
sg('set_kernel_optimization_type', 'FASTBUTMEMHUNGRY');
ktref=sg('get_kernel_matrix', 'TEST');
sg('new_classifier', 'SVMLIGHT');
sg('c', C);
tic; sg(train_classifier'); tref=toc
[bref, alphasref]=sg('get_svm');
tic;
sg('init_kernel_optimization');
outoptref=sg('classify');
toutref=toc

outopt(1:10)
outoptref(1:10)
max(abs(outopt-outoptref))

t
tref
tout
toutref
b
bref
