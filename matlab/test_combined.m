XT1=rand(120,1000);
XT2=rand(10,1000);
XT3=rand(5,1000);

LT=sign(rand(1,1000)-0.5);

gf('set_labels', 'TRAIN', LT);
gf('add_features', 'TRAIN', XT1);
gf('add_features', 'TRAIN', XT2);
gf('add_features', 'TRAIN', XT3);

%combined kernel with 200 mb cache for the comb. kernel
gf('send_command', 'set_kernel COMBINED 200');

%other kernels (linear 10 mb cache, gaussian 20 mb cache sigma 1, poly 50mb
%cache degreee 3 homogene
gf('send_command', 'add_kernel LINEAR REAL 10 ');
gf('send_command', 'add_kernel GAUSSIAN REAL 20 1');
gf('send_command', 'add_kernel POLY REAL 50 3 0');

gf('send_command', 'init_kernel TRAIN');
gf('send_command', 'c 5');
gf('send_command', 'new_svm LIGHT');
gf('send_command', 'svm_train');
