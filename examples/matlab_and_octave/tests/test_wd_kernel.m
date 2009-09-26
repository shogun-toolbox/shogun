C=1;
order=20;
order_com=5;
max_mismatch=0;
len=200;
shift=0;
num=100;
num_test=5000;
cache=10;
normalize=1;
mkl_stepsize=1;
block=0;
single_degree=-1;

acgt='ACGT';
rand('state',1);
traindat=acgt(ceil(4*rand(len,num)));
trainlab=[-ones(1,num/2),ones(1,num/2)];

testdat=acgt(ceil(4*rand(len,num_test)));
testlab=[-ones(1,num/2),ones(1,num_test/2)];
x=ceil(linspace(0,shift,len));
shifts = int32(x(end:-1:1));

%sg('loglevel', 'ALL');
sg('clean_features', 'TRAIN');
sg('clean_features', 'TEST');
sg('clean_kernel');
sg('use_linadd', 1);
sg('use_batch_computation', 1);

sg('set_features', 'TRAIN', traindat, 'DNA');
sg('set_labels', 'TRAIN', trainlab);

sg('set_features', 'TEST', testdat, 'DNA');
sg('set_labels', 'TEST', testlab);
%
sg('set_kernel', 'WEIGHTEDDEGREE', 'CHAR', cache, order, max_mismatch, normalize, mkl_stepsize, block, single_degree);
kt=sg('get_kernel_matrix', 'TRAIN');
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
sg('use_linadd', 1);
sg('use_batch_computation', 1);

sg('set_features', 'TRAIN', traindat, 'DNA');
sg('set_labels', 'TRAIN', trainlab);
%
sg('set_features', 'TEST', testdat, 'DNA');
sg('set_labels', 'TEST', testlab);
%
sg('set_kernel', 'WEIGHTEDDEGREEPOS2', 'CHAR', 10, order, max_mismatch, len, shifts);
ktref=sg('get_kernel_matrix', 'TRAIN');
sg('new_classifier', 'SVMLIGHT');
sg('c', C);
tic; sg('train_classifier'); tref=toc
[bref, alphasref]=sg('get_svm');
tic;
outoptref=sg('classify');
toutref=toc

outopt(1:10)
outoptref(1:10)
max(abs(outopt-outoptref))

t
tref
tout
toutref
