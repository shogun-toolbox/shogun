C=1;
order=10 ;
mismatch=0;
len=200;
shift=10;
num=20;
num_test=200;
cache=10;

acgt='ACGT';
rand('state',1);
traindat1=acgt(ceil(4*rand(len,num)));
trainlab=[-ones(1,num/2),ones(1,num/2)];
traindat1(1:1, trainlab==1)='A' ;

testdat1=acgt(ceil(4*rand(len,num_test)));
testlab=[-ones(1,num_test/2),ones(1,num_test/2)];
testdat1(1:1, testlab==1)='A' ;

x=ceil(linspace(0,shift,len));
shifts = sprintf( '%i ', x(end:-1:1) );

sg('send_command', 'loglevel ALL');
sg('send_command','clean_features TRAIN');
sg('send_command','clean_features TEST');
sg('send_command','clean_kernels');
sg('send_command', 'use_linadd 0' );
sg('send_command', 'use_batch_computation 0');

sg('set_features', 'TRAIN', traindat1, 'DNA');
sg('set_labels', 'TRAIN', trainlab);

sg('set_features', 'TEST', testdat1,'DNA');
%
%sg('send_command', sprintf( 'set_kernel WEIGHTEDDEGREEPOS3 CHAR 10 %i %i %i 1 %s', order, mismatch, len, shifts ) );
sg('send_command', sprintf( 'set_kernel WEIGHTEDDEGREEPOSPHYL3 CHAR 10 %i %i %i 1 %s', order, mismatch, len, shifts ) );

sg('send_command', 'set_kernel_optimization_type FASTBUTMEMHUNGRY' );
%sg('send_command', 'set_kernel_optimization_type SLOWBUTMEMEFFICIENT' );
sg('send_command', 'init_kernel TRAIN');
sg('set_WD_position_weights', rand(1,len)) ;
%sg('set_subkernel_weights', rand(len, num)) ;

%kt=sg('get_kernel_matrix');
sg('send_command', 'new_svm LIGHT');
sg('send_command', sprintf('c %f',C));
tic; sg('send_command', 'svm_train'); t=toc
[b, alphas]=sg('get_svm');

tic;
sg('send_command', 'init_kernel TEST');
%sg('set_subkernel_weights', rand(len,num_test)) ;

sg('send_command', 'set_kernel_optimization_type FASTBUTMEMHUNGRY' );
sg('send_command', 'use_linadd 1' );
sg('send_command', 'init_kernel_optimization')
out1=sg('svm_classify');
sg('send_command', 'delete_kernel_optimization')

sg('send_command', 'set_kernel_optimization_type SLOWBUTMEMEFFICIENT' );
sg('send_command', 'use_linadd 1' );
sg('send_command', 'init_kernel_optimization')
out2=sg('svm_classify');
sg('send_command', 'delete_kernel_optimization')

sg('send_command', 'use_linadd 0' );
out3=sg('svm_classify');

sg('send_command', 'use_batch_computation 1');
out4=sg('svm_classify');

tout=toc
