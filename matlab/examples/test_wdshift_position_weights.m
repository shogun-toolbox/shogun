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
trainw=rand(size(traindat)) ;
trainlab=[-ones(1,num/2),ones(1,num/2)];

testdat=acgt(ceil(4*rand(len,num_test)));
testw=rand(size(testdat)) ;
testlab=[-ones(1,num/2),ones(1,num_test/2)];

x=ceil(linspace(1,shift,len));
shifts = sprintf( '%i ', x(end:-1:1) );

sg('send_command', 'loglevel ALL');
sg('send_command','clean_features TRAIN');
sg('send_command','clean_features TEST');
sg('send_command','clean_kernels');
sg('send_command', 'use_linadd 0' );                  % important--other cases not implemented
sg('send_command', 'use_batch_computation 0');        % important--other cases not implemented

sg('set_features', 'TRAIN', traindat, 'DNA');
sg('set_labels', 'TRAIN', trainlab);

% use WEIGHTEDDEGREEPOS3 without normalization -- otherwise, you might get wrong results
sg('send_command', sprintf( 'set_kernel WEIGHTEDDEGREEPOS3 CHAR 10 %i %i %i 1 %s', order, mismatch, len, shifts ) );

% first initialize
sg('send_command', 'init_kernel TRAIN');

% then set weights
sg('set_WD_position_weights', trainw, 'TRAIN') ;
sg('set_WD_position_weights', trainw, 'TEST') ;

% train the svm
sg('send_command', 'new_svm LIGHT');
sg('send_command', sprintf('c %f',C));
tic; sg('send_command', 'svm_train'); t=toc
[b, alphas]=sg('get_svm');

% set features and initialize
sg('set_features', 'TEST', testdat, 'DNA');
sg('send_command', 'init_kernel TEST');

% change rhs of weights
sg('set_WD_position_weights', testw, 'TEST') ;

% classify
outopt=sg('svm_classify');
tout=toc

