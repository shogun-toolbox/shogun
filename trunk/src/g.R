#

dyn.load("sg.so")
#a <- .External("sg","test", "blah blub")
#b <- .External("sg","test", "blah blub", c(1.0,2.0))
#c <- .External("sg","test", "blah 1 2 3", c(1.2,1.0,2.0))
#d <- .External("sg","test", "bla ba")
#.External("sg","sg", "help")
#.External("sg","sg", "send_command", "loglevel ALL")
mat <- array(1:20, dim=c(4,10))
mat2 <- matrix(33.77, 10, 20)

#.External("sg","matrix_test", mat, c(4,10))
#.External("sg","matrix_test", mat)

# Matlab example

C <- 1;
order <- 20;
order_com <- 5;
mismatch <- 0;
len <- 200;
shift <- 32;
#2.7101e+03
num <- 100;
num_test <- 5000;
cache <- 10;

#acgt="ACGT";
#rand("state",1);
#traindat1=acgt(ceil(4*rand(len,num)));
#traindat2=acgt(ceil(4*rand(len,num)));
#traindat3=rand(len,num);
#traindat4=rand(len,num);
#trainlab=[-ones(1,num/2),ones(1,num/2)];

#testdat1=acgt(ceil(4*rand(len,num_test)));
#testdat2=acgt(ceil(4*rand(len,num_test)));
#testdat3=rand(len,num_test);
#testdat4=rand(len,num_test);
#testlab=[-ones(1,num/2),ones(1,num_test/2)];
#shifts = sprintf( "#i ", shift * ones(1,len) );

.External("sg","send_command", "loglevel ALL");
.External("sg","send_command","clean_features TRAIN");
.External("sg","send_command","clean_features TEST");
.External("sg","send_command","clean_kernels");
.External("sg","send_command", "use_linadd 1" );


traindat1 <- matrix(0.0, 4, 5)
traindat2 <- matrix(0.0, 4, 5)

.External("sg","add_features", "TRAIN", traindat1);
.External("sg","add_features", "TRAIN", traindat2);
.External("sg","send_command", "add_preproc LOGPLUSONE");
.External("sg","send_command", "add_preproc LOGPLUSONE");
.External("sg","send_command", "add_preproc PRUNEVARSUBMEAN");
.External("sg","send_command", "attach_preproc TRAIN");


.External("sg","add_features", "TEST", testdat1);
.External("sg","add_features", "TEST", testdat3);
.External("sg","send_command", "attach_preproc TEST");

#.External("sg","add_features", "TRAIN", traindat4);
#.External("sg","add_features", "TRAIN", traindat2);
.External("sg","send_command", sprintf("convert TRAIN SIMPLE CHAR SIMPLE WORD DNA %i %i", order_com, order_com-1 ) );
.External("sg","send_command", "clean_preproc" );
.External("sg","send_command", "add_preproc SORTWORD");
.External("sg","send_command", "attach_preproc TRAIN");
.External("sg","set_labels", "TRAIN", trainlab);

#.External("sg","add_features", "TEST", testdat4);
#.External("sg","add_features", "TEST", testdat2);
#.External("sg","send_command", sprintf("convert TEST SIMPLE CHAR SIMPLE WORD DNA %i %i", order_com, order_com-1 ) );
#.External("sg","set_labels", "TEST", testlab);
#.External("sg","send_command", "attach_preproc TEST");
#%
#.External("sg","send_command", sprintf( "set_kernel COMBINED %i", cache) );
#.External("sg","send_command", sprintf( "add_kernel 1.0 WEIGHTEDDEGREEPOS3 CHAR 10 %i %i %i 1 %s", order, mismatch, len, shifts ) );
#.External("sg","send_command", sprintf( "add_kernel 1.0 LINEAR REAL 10 1.0" ) );
#%.External("sg","send_command", sprintf( "add_kernel 4.0 GAUSSIAN REAL 10 1.0" ) );
#.External("sg","send_command", sprintf( "add_kernel 1.0 COMM WORD 10 0" ) );
#.External("sg","send_command", "init_kernel TRAIN");
#
#
#%.External("sg","send_command", "set_kernel_optimization_type FASTBUTMEMHUNGRY" );
#.External("sg","send_command", "set_kernel_optimization_type SLOWBUTMEMEFFICIENT" );
#%kt=.External("sg","get_kernel_matrix");
#
#.External("sg","send_command", "new_svm LIGHT");
#.External("sg","send_command", "delete_kernel_optimization");
#%.External("sg","send_command", "init_kernel_optimization");
#.External("sg","send_command", sprintf("c %f",C));
#tic; .External("sg","send_command", "svm_train"); t=toc
#[b, alphas]=.External("sg","get_svm");
#
#%.External("sg","set_features", "TEST", testdat);
#%.External("sg","set_labels", "TEST", testlab);
#%.External("sg","send_command", "init_kernel TEST");
#%%kte=.External("sg","get_kernel_matrix");
#%out=.External("sg","svm_classify");
#%
#%
#tic;
#%.External("sg","send_command", "set_kernel_optimization_type SLOWBUTMEMEFFICIENT" );
#%.External("sg","send_command", "set_kernel_optimization_type FASTBUTMEMHUNGRY" );
#.External("sg","send_command", "delete_kernel_optimization");
#%.External("sg","send_command", "init_kernel_optimization");
#.External("sg","send_command", "init_kernel TEST");
#outopt=.External("sg","svm_classify");
#tout=toc
#.External("sg","send_command", "delete_kernel_optimization");
#
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
#sgref2("send_command", "loglevel ALL");
#sgref2("send_command","clean_features TRAIN");
#sgref2("send_command","clean_features TEST");
#sgref2("send_command","clean_kernels");
#sgref2("send_command", "use_linadd 1" );
#
#sgref2("add_features", "TRAIN", traindat1);
#sgref2("add_features", "TRAIN", traindat2);
#sgref2("send_command", sprintf("convert TRAIN SIMPLE CHAR SIMPLE WORD DNA %i %i", order_com, order_com-1 ) );
#sgref2("send_command", "clean_preproc" );
#sgref2("send_command", "add_preproc SORTWORD");
#sgref2("send_command", "attach_preproc TRAIN");
#sgref2("add_features", "TRAIN", traindat3);
#sgref2("send_command", "add_preproc LOGPLUSONE");
#sgref2("send_command", "add_preproc LOGPLUSONE");
#sgref2("send_command", "add_preproc PRUNEVARSUBMEAN");
#sgref2("send_command", "attach_preproc TRAIN");
#%sgref2("add_features", "TRAIN", traindat4);
#sgref2("set_labels", "TRAIN", trainlab);
#%
#sgref2("set_features", "TEST", testdat1);
#sgref2("add_features", "TEST", testdat2);
#sgref2("send_command", sprintf("convert TEST SIMPLE CHAR SIMPLE WORD DNA %i %i", order_com, order_com-1 ) );
#sgref2("add_features", "TEST", testdat3);
#%sgref2("add_features", "TEST", testdat4);
#sgref2("set_labels", "TEST", testlab);
#sgref2("send_command", "attach_preproc TEST");
#%
#sgref2("send_command", sprintf( "set_kernel COMBINED %i", cache) );
#sgref2("send_command", sprintf( "add_kernel 1.0 WEIGHTEDDEGREEPOS3 CHAR 10 %i %i %i 1 %s", order, mismatch, len, shifts ) );
#sgref2("send_command", sprintf( "add_kernel 1.0 COMM WORD 10 0" ) );
#sgref2("send_command", sprintf( "add_kernel 1.0 LINEAR REAL 10 1.0" ) );
#%sgref2("send_command", sprintf( "add_kernel 4.0 GAUSSIAN REAL 10 1.0" ) );
#
#%ktref=sgref2("get_kernel_matrix");
#
#sgref2("send_command", "new_svm LIGHT");
#sgref2("send_command", sprintf("c %f",C));
#sgref2("send_command", "init_kernel TRAIN");
#tic; sgref2("send_command", "svm_train"); tref=toc
#[bref, alphasref]=sgref2("get_svm");
#%sgref2("set_svm",b,alphas);
#sgref2("send_command", "init_kernel_optimization");
#%sgref2("send_command", "delete_kernel_optimization");
#%sgref2("set_features", "TEST", testdat);
#%sgref2("set_labels", "TEST", testlab);
#%sgref2("send_command", "init_kernel TEST");
#%%kteref=sgref2("get_kernel_matrix");
#%outref=sgref2("svm_classify");
#%
#%
#tic;
#sgref2("send_command", "init_kernel TEST");
#outoptref=sgref2("svm_classify");
#toutref=toc
#sgref2("send_command", "delete_kernel_optimization");
#
#%out(1:10)
#outopt(1:10)
#%outref(1:10)
#outoptref(1:10)
#max(abs(outopt-outoptref))
#
#tout
###############################################################################################toutref
