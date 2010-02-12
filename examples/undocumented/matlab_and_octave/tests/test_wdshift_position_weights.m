C=1;
order=20;
order_com=5;
mismatch=0;
len=200;
shift=100;
num=100;
num_test=5000;
cache=10;

acgt='ACGT';
rand('state',1);
traindat=acgt(ceil(4*rand(len,num)));
trainw=rand(size(traindat(1:len)));
trainlab=[-ones(1,num/2),ones(1,num/2)];

testdat=acgt(ceil(4*rand(len,num_test)));
testw=rand(size(testdat(1:len)));
testlab=[-ones(1,num/2),ones(1,num_test/2)];

x=ceil(linspace(1,shift,len));
shifts = int32(x(end:-1:1));

%sg('loglevel', 'ALL');
sg('clean_features', 'TRAIN');
sg('clean_features', 'TEST');
sg('clean_kernel');
sg('use_linadd',  0);                  % important--other cases not implemented
sg('use_batch_computation', 0);        % important--other cases not implemented

sg('set_features', 'TRAIN', traindat, 'DNA');
sg('set_labels', 'TRAIN', trainlab);

% use WEIGHTEDDEGREEPOS3 without normalization -- otherwise, you might get wrong results
mkl_stepsize=1;
sg('set_kernel', 'WEIGHTEDDEGREEPOS3', 'CHAR', 10', order, mismatch, len, mkl_stepsize, shifts);

% first initialize

% then set weights
sg('set_WD_position_weights', trainw, 'TRAIN') ;
sg('set_WD_position_weights', trainw, 'TEST') ;

% train the svm
sg('new_classifier', 'SVMLIGHT');
sg('c',C);
tic; sg('train_classifier'); t=toc
[b, alphas]=sg('get_svm');

% set features and initialize
sg('set_features', 'TEST', testdat, 'DNA');

% change rhs of weights
sg('set_WD_position_weights', testw, 'TEST') ;

% classify
outopt=sg('classify');
tout=toc


