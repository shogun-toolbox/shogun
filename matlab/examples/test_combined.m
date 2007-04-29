XT1=rand(120,1000);
XT2=rand(10,1000);
XT3=rand(5,1000);

LT=sign(rand(1,1000)-0.5);

sg('send_command', 'clean_features TRAIN');
sg('send_command', 'clean_features TEST');
sg('send_command', 'clean_kernels');

sg('set_labels', 'TRAIN', LT);
sg('add_features', 'TRAIN', XT1);
sg('add_features', 'TRAIN', XT2);
sg('add_features', 'TRAIN', XT3);

%combined kernel with 200 mb cache for the comb. kernel
sg('send_command', 'set_kernel COMBINED 200');

%other kernels (linear 10 mb cache, gaussian 20 mb cache sigma 1, poly 50mb
%cache degreee 3 homogene
sg('send_command', 'add_kernel 1 LINEAR REAL 10 ');
sg('send_command', 'add_kernel 1 GAUSSIAN REAL 20 1');
sg('send_command', 'add_kernel 1 POLY REAL 50 3 0');

sg('send_command', 'init_kernel TRAIN');
sg('send_command', 'c 5');
sg('send_command', 'new_svm LIGHT');
sg('send_command', 'svm_train');


Ks=sg('get_kernel_matrix') ;
