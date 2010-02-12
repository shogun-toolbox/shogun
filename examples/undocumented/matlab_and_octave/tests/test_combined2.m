XT1=rand(120,1000);
XT2=rand(10,1000);
XT3=rand(5,1000);

LT=sign(rand(1,1000)-0.5);

sg('clean_features', 'TRAIN');
sg('clean_features', 'TEST');
sg('clean_kernel');

sg('set_labels', 'TRAIN', LT);
sg('add_features', 'TRAIN', XT1);
sg('add_features', 'TRAIN', XT2);
sg('add_features', 'TRAIN', XT3);
sg('add_features', 'TEST', XT1);
sg('add_features', 'TEST', XT2);
sg('add_features', 'TEST', XT3);

%combined kernel with 200 mb cache for the comb. kernel
sg('set_kernel', 'COMBINED', 200);

%sg('add_kernel', 1, 'CUSTOM', 'ANY', 50);
%sg('set_custom_kernel',kt,'FULL2DIAG');
%other kernels (linear 10 mb cache, gaussian 20 mb cache sigma 1, poly 50mb
%cache degreee 3 homogene
sg('add_kernel', 1, 'LINEAR', 'REAL', 10);
sg('add_kernel', 2, 'GAUSSIAN', 'REAL', 20, 1);
sg('add_kernel', 3, 'POLY', 'REAL', 50, 3, 0);

sg('c', 5);
sg('new_classifier', 'SVMLIGHT');
sg('train_classifier');
trKs=sg('get_kernel_matrix', 'TRAIN') ;
Ks=sg('get_kernel_matrix', 'TEST') ;

sg('set_features', 'TRAIN', XT1);
sg('set_features', 'TEST', XT1);
sg('set_kernel', 'LINEAR', 'REAL', 10);
trK1=sg('get_kernel_matrix', 'TRAIN') ;
K1=sg('get_kernel_matrix', 'TEST') ;


sg('set_features', 'TRAIN', XT2);
sg('set_features', 'TEST', XT2);
sg('set_kernel', 'GAUSSIAN', 'REAL', 20, 1);
trK2=sg('get_kernel_matrix', 'TRAIN') ;
K2=sg('get_kernel_matrix', 'TEST') ;



sg('set_features', 'TRAIN', XT3);
sg('set_features', 'TEST', XT3);
sg('set_kernel', 'POLY', 'REAL', 50, 3, 0);
trK3=sg('get_kernel_matrix', 'TRAIN') ;
K3=sg('get_kernel_matrix', 'TEST') ;


norm(trKs-1*trK1-2*trK2-3*trK3)
norm(Ks-1*K1-2*K2-3*K3)
