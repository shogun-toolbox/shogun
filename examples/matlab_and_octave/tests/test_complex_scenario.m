C=1;
order=20;
order_com=5;
mismatch=0;
len=200;
shift=0;
num=100;
num_test=200;
cache=10;
max_mismatch=0;
normalize=1;
mkl_stepsize=1;
block=0;
single_degree=-1;

acgt='ACGT';
rand('state',1);
traindat1=acgt(ceil(4*rand(len,num)));
traindat2=acgt(ceil(4*rand(len,num)));
traindat3=rand(len,num);
traindat4=rand(len,num);
trainlab=[-ones(1,num/2),ones(1,num/2)];

testdat1=acgt(ceil(4*rand(len,num_test)));
testdat2=acgt(ceil(4*rand(len,num_test)));
testdat3=rand(len,num_test);
testdat4=rand(len,num_test);
testlab=[-ones(1,num/2),ones(1,num_test/2)];
x=ceil(linspace(1,shift,len));
shifts = int32(x(end:-1:1));

%sg('loglevel', 'ALL');
sg('clean_features', 'TRAIN');
sg('clean_features', 'TEST');
sg('clean_kernel');
sg('threads', 4);
sg('use_linadd', 1);
sg('use_batch_computation', 1);

sg('add_features', 'TRAIN', traindat1, 'DNA');
sg('add_features', 'TRAIN', traindat2, 'DNA');
sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', 'WORD', order_com, order_com-1);
sg('clean_preproc');
sg('add_preproc', 'SORTWORDSTRING');
sg('attach_preproc', 'TRAIN');
sg('add_features', 'TRAIN', traindat3);
sg('add_preproc', 'LOGPLUSONE');
sg('add_preproc', 'LOGPLUSONE');
sg('add_preproc', 'PRUNEVARSUBMEAN');
sg('attach_preproc', 'TRAIN');
sg('add_features', 'TRAIN', traindat4);
sg('set_labels', 'TRAIN', trainlab);

sg('add_features', 'TEST', testdat1, 'DNA');
sg('add_features', 'TEST', testdat2, 'DNA');
sg('convert', 'TEST', 'STRING', 'CHAR', 'STRING', 'WORD', order_com, order_com-1);
sg('add_features', 'TEST', testdat3);
sg('add_features', 'TEST', testdat4);
sg('set_labels', 'TEST', testlab);
sg('attach_preproc', 'TEST');
%
sg('set_kernel', 'COMBINED', cache);
%sg('add_kernel', 1.0, 'WEIGHTEDDEGREE', 'CHAR', cache, order, max_mismatch, normalize, mkl_stepsize, block, single_degree);
sg('add_kernel', 1.0, 'WEIGHTEDDEGREEPOS2', 'CHAR', 10', order, mismatch, len, shifts);
sg('add_kernel', 1.0, 'COMMSTRING', 'WORD', 10, 0);
sg('add_kernel', 1.0, 'LINEAR', 'REAL', 10, 1.0);
sg('add_kernel', 4.0, 'GAUSSIAN', 'REAL', 10, 1.0);
sg('set_kernel_optimization_type', 'FASTBUTMEMHUNGRY');
%sg('set_kernel_optimization_type', 'SLOWBUTMEMEFFICIENT');
kt=sg('get_kernel_matrix', 'TRAIN');
sg('new_classifier', 'SVMLIGHT');
sg('c', C);
tic; sg('train_classifier'); t=toc
[b, alphas]=sg('get_svm');

tic;
kte=sg('get_kernel_matrix', 'TEST');
outopt=sg('classify');
tout=toc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%sg('loglevel', 'ALL');
sg('clean_features', 'TRAIN');
sg('clean_features', 'TEST');
sg('threads', 4);
sg('clean_kernel');
sg('use_linadd', 0);
sg('use_batch_computation', 0);

sg('add_features', 'TRAIN', traindat1, 'DNA');
sg('add_features', 'TRAIN', traindat2, 'DNA');
sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', 'WORD', order_com, order_com-1);
sg('clean_preproc');
sg('add_preproc', 'SORTWORDSTRING');
sg('attach_preproc', 'TRAIN');
sg('add_features', 'TRAIN', traindat3);
sg('add_preproc', 'LOGPLUSONE');
sg('add_preproc', 'LOGPLUSONE');
sg('add_preproc', 'PRUNEVARSUBMEAN');
sg('attach_preproc', 'TRAIN');
sg('add_features', 'TRAIN', traindat4);
sg('set_labels', 'TRAIN', trainlab);
%
sg('add_features', 'TEST', testdat1, 'DNA');
sg('add_features', 'TEST', testdat2, 'DNA');
sg('convert', 'TEST', 'STRING', 'CHAR', 'STRING', 'WORD', order_com, order_com-1);
sg('add_features', 'TEST', testdat3);
sg('add_features', 'TEST', testdat4);
sg('set_labels', 'TEST', testlab);
sg('attach_preproc', 'TEST');
%
sg('set_kernel', 'COMBINED', cache);
%sg('add_kernel', 1.0, 'WEIGHTEDDEGREE', 'CHAR', cache, order, max_mismatch, normalize, mkl_stepsize, block, single_degree);
sg('add_kernel', 1.0, 'WEIGHTEDDEGREEPOS2', 'CHAR', 10, order, mismatch, len, shifts);
sg('add_kernel', 1.0, 'COMMSTRING', 'WORD', 10, 0);
sg('add_kernel', 1.0, 'LINEAR', 'REAL', 10, 1.0);
sg('add_kernel', 4.0, 'GAUSSIAN', 'REAL', 10, 1.0);
%sg('set_kernel_optimization_type', 'FASTBUTMEMHUNGRY');
sg('set_kernel_optimization_type', 'SLOWBUTMEMEFFICIENT');
ktref=sg('get_kernel_matrix', 'TRAIN');
sg('new_classifier', 'SVMLIGHT');
sg('c', C);
tic; sg('train_classifier'); tref=toc
[bref, alphasref]=sg('get_svm');
tic;
sg('init_kernel_optimization');
kteref=sg('get_kernel_matrix', 'TEST');
outoptref=sg('classify');
toutref=toc

outopt(1:10)
outoptref(1:10)
max(abs(outopt-outoptref))

t
tref
tout
toutref
