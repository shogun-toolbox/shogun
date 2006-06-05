C=1;
order=20;
order_com=5;
mismatch=0;
len=200;
shift=32;
%2.7101e+03
num=100;
num_test=5000;
cache=10;

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
shifts = sprintf( '%i ', x(end:-1:1) );

gf('send_command', 'loglevel ALL');
gf('send_command','clean_features TRAIN');
gf('send_command','clean_features TEST');
gf('send_command','clean_kernels');
gf('send_command', 'use_linadd 1' );

gf('add_features', 'TRAIN', traindat1);
gf('add_features', 'TRAIN', traindat3);
gf('send_command', 'add_preproc LOGPLUSONE');
gf('send_command', 'add_preproc LOGPLUSONE');
gf('send_command', 'add_preproc PRUNEVARSUBMEAN');
gf('send_command', 'attach_preproc TRAIN');

gf('add_features', 'TEST', testdat1);
gf('add_features', 'TEST', testdat3);
gf('send_command', 'attach_preproc TEST');

%gf('add_features', 'TRAIN', traindat4);
gf('add_features', 'TRAIN', traindat2);
gf('send_command', sprintf('convert TRAIN SIMPLE CHAR SIMPLE WORD DNA %i %i', order_com, order_com-1 ) );
gf('send_command', 'clean_preproc' );
gf('send_command', 'add_preproc SORTWORD');
gf('send_command', 'attach_preproc TRAIN');
gf('set_labels', 'TRAIN', trainlab);
%
%gf('add_features', 'TEST', testdat4);
gf('add_features', 'TEST', testdat2);
gf('send_command', sprintf('convert TEST SIMPLE CHAR SIMPLE WORD DNA %i %i', order_com, order_com-1 ) );
gf('set_labels', 'TEST', testlab);
gf('send_command', 'attach_preproc TEST');
%
gf('send_command', sprintf( 'set_kernel COMBINED %i', cache) );
gf('send_command', sprintf( 'add_kernel 1.0 WEIGHTEDDEGREEPOS3 CHAR 10 %i %i %i 1 %s', order, mismatch, len, shifts ) );
gf('send_command', sprintf( 'add_kernel 1.0 LINEAR REAL 10 1.0' ) );
%gf('send_command', sprintf( 'add_kernel 4.0 GAUSSIAN REAL 10 1.0' ) );
gf('send_command', sprintf( 'add_kernel 1.0 COMM WORD 10 0' ) );
gf('send_command', 'init_kernel TRAIN');


%gf('send_command', 'set_kernel_optimization_type FASTBUTMEMHUNGRY' );
gf('send_command', 'set_kernel_optimization_type SLOWBUTMEMEFFICIENT' );
gf('send_command', 'use_batch_computation 0');
%kt=gf('get_kernel_matrix');

gf('send_command', 'new_svm LIGHT');
%gf('send_command', 'delete_kernel_optimization');
%gf('send_command', 'init_kernel_optimization');
gf('send_command', sprintf('c %f',C));
tic; gf('send_command', 'svm_train'); t=toc
[b, alphas]=gf('get_svm');

%gf('set_features', 'TEST', testdat);
%gf('set_labels', 'TEST', testlab);
%gf('send_command', 'init_kernel TEST');
%%kte=gf('get_kernel_matrix');
%out=gf('svm_classify');
%
%
tic;
%gf('send_command', 'set_kernel_optimization_type SLOWBUTMEMEFFICIENT' );
%gf('send_command', 'set_kernel_optimization_type FASTBUTMEMHUNGRY' );
%gf('send_command', 'delete_kernel_optimization');
%gf('send_command', 'init_kernel_optimization');
gf('send_command', 'init_kernel TEST');
outopt=gf('svm_classify');
tout=toc
%gf('send_command', 'delete_kernel_optimization');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

gf('send_command', 'loglevel ALL');
gf('send_command','clean_features TRAIN');
gf('send_command','clean_features TEST');
gf('send_command','clean_kernels');
gf('send_command', 'use_linadd 1' );

gf('add_features', 'TRAIN', traindat1);
gf('add_features', 'TRAIN', traindat2);
gf('send_command', sprintf('convert TRAIN SIMPLE CHAR SIMPLE WORD DNA %i %i', order_com, order_com-1 ) );
gf('send_command', 'clean_preproc' );
gf('send_command', 'add_preproc SORTWORD');
gf('send_command', 'attach_preproc TRAIN');
gf('add_features', 'TRAIN', traindat3);
gf('send_command', 'add_preproc LOGPLUSONE');
gf('send_command', 'add_preproc LOGPLUSONE');
gf('send_command', 'add_preproc PRUNEVARSUBMEAN');
gf('send_command', 'attach_preproc TRAIN');
%gf('add_features', 'TRAIN', traindat4);
gf('set_labels', 'TRAIN', trainlab);
%
gf('set_features', 'TEST', testdat1);
gf('add_features', 'TEST', testdat2);
gf('send_command', sprintf('convert TEST SIMPLE CHAR SIMPLE WORD DNA %i %i', order_com, order_com-1 ) );
gf('add_features', 'TEST', testdat3);
%gf('add_features', 'TEST', testdat4);
gf('set_labels', 'TEST', testlab);
gf('send_command', 'attach_preproc TEST');
%
gf('send_command', sprintf( 'set_kernel COMBINED %i', cache) );
gf('send_command', sprintf( 'add_kernel 1.0 WEIGHTEDDEGREEPOS3 CHAR 10 %i %i %i 1 %s', order, mismatch, len, shifts ) );
gf('send_command', sprintf( 'add_kernel 1.0 COMM WORD 10 0' ) );
gf('send_command', sprintf( 'add_kernel 1.0 LINEAR REAL 10 1.0' ) );
%gf('send_command', sprintf( 'add_kernel 4.0 GAUSSIAN REAL 10 1.0' ) );
gf('send_command', 'set_kernel_optimization_type FASTBUTMEMHUNGRY' );
%gf('send_command', 'set_kernel_optimization_type SLOWBUTMEMEFFICIENT' );

%ktref=gf('get_kernel_matrix');

gf('send_command', 'new_svm LIGHT');
gf('send_command', sprintf('c %f',C));
gf('send_command', 'init_kernel TRAIN');
tic; gf('send_command', 'svm_train'); tref=toc
[bref, alphasref]=gf('get_svm');
%gf('set_svm',b,alphas);
%gf('send_command', 'init_kernel_optimization');
%gf('send_command', 'delete_kernel_optimization');
%gf('set_features', 'TEST', testdat);
%gf('set_labels', 'TEST', testlab);
%gf('send_command', 'init_kernel TEST');
%%kteref=gf('get_kernel_matrix');
%outref=gf('svm_classify');
%
%
tic;
gf('send_command', 'use_batch_computation 0');
gf('send_command', 'init_kernel TEST');
outoptref=gf('svm_classify');
toutref=toc
%gf('send_command', 'delete_kernel_optimization');

%out(1:10)
outopt(1:10)
%outref(1:10)
outoptref(1:10)
max(abs(outopt-outoptref))

t
tref
tout
toutref
