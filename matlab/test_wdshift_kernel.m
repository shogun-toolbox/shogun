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
shifts = sprintf( '%i ', shift * ones(1,len) );

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
%kt=gf('get_kernel_matrix');

gf('send_command', 'new_svm LIGHT');
gf('send_command', 'delete_kernel_optimization');
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
gf('send_command', 'delete_kernel_optimization');
%gf('send_command', 'init_kernel_optimization');
gf('send_command', 'init_kernel TEST');
outopt=gf('svm_classify');
tout=toc
gf('send_command', 'delete_kernel_optimization');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

gfref2('send_command', 'loglevel ALL');
gfref2('send_command','clean_features TRAIN');
gfref2('send_command','clean_features TEST');
gfref2('send_command','clean_kernels');
gfref2('send_command', 'use_linadd 1' );

gfref2('add_features', 'TRAIN', traindat1);
gfref2('add_features', 'TRAIN', traindat2);
gfref2('send_command', sprintf('convert TRAIN SIMPLE CHAR SIMPLE WORD DNA %i %i', order_com, order_com-1 ) );
gfref2('send_command', 'clean_preproc' );
gfref2('send_command', 'add_preproc SORTWORD');
gfref2('send_command', 'attach_preproc TRAIN');
gfref2('add_features', 'TRAIN', traindat3);
gfref2('send_command', 'add_preproc LOGPLUSONE');
gfref2('send_command', 'add_preproc LOGPLUSONE');
gfref2('send_command', 'add_preproc PRUNEVARSUBMEAN');
gfref2('send_command', 'attach_preproc TRAIN');
%gfref2('add_features', 'TRAIN', traindat4);
gfref2('set_labels', 'TRAIN', trainlab);
%
gfref2('set_features', 'TEST', testdat1);
gfref2('add_features', 'TEST', testdat2);
gfref2('send_command', sprintf('convert TEST SIMPLE CHAR SIMPLE WORD DNA %i %i', order_com, order_com-1 ) );
gfref2('add_features', 'TEST', testdat3);
%gfref2('add_features', 'TEST', testdat4);
gfref2('set_labels', 'TEST', testlab);
gfref2('send_command', 'attach_preproc TEST');
%
gfref2('send_command', sprintf( 'set_kernel COMBINED %i', cache) );
gfref2('send_command', sprintf( 'add_kernel 1.0 WEIGHTEDDEGREEPOS3 CHAR 10 %i %i %i 1 %s', order, mismatch, len, shifts ) );
gfref2('send_command', sprintf( 'add_kernel 1.0 COMM WORD 10 0' ) );
gfref2('send_command', sprintf( 'add_kernel 1.0 LINEAR REAL 10 1.0' ) );
%gfref2('send_command', sprintf( 'add_kernel 4.0 GAUSSIAN REAL 10 1.0' ) );

%ktref=gfref2('get_kernel_matrix');

gfref2('send_command', 'new_svm LIGHT');
gfref2('send_command', sprintf('c %f',C));
gfref2('send_command', 'init_kernel TRAIN');
tic; gfref2('send_command', 'svm_train'); tref=toc
[bref, alphasref]=gfref2('get_svm');
%gfref2('set_svm',b,alphas);
gfref2('send_command', 'init_kernel_optimization');
%gfref2('send_command', 'delete_kernel_optimization');
%gfref2('set_features', 'TEST', testdat);
%gfref2('set_labels', 'TEST', testlab);
%gfref2('send_command', 'init_kernel TEST');
%%kteref=gfref2('get_kernel_matrix');
%outref=gfref2('svm_classify');
%
%
tic;
gfref2('send_command', 'init_kernel TEST');
outoptref=gfref2('svm_classify');
toutref=toc
gfref2('send_command', 'delete_kernel_optimization');

%out(1:10)
outopt(1:10)
%outref(1:10)
outoptref(1:10)
max(abs(outopt-outoptref))

tout
toutref
