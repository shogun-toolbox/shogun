XT1=rand(120,1000);
XT2=rand(10,1000);
XT3=rand(5,1000);

LT=sign(rand(1,1000)-0.5);

gf_('set_labels', 'TRAIN', LT);
gf_('add_features', 'TRAIN', XT1);
gf_('add_features', 'TRAIN', XT2);
gf_('add_features', 'TRAIN', XT3);
gf_('add_features', 'TEST', XT1);
gf_('add_features', 'TEST', XT2);
gf_('add_features', 'TEST', XT3);

%combined kernel with 200 mb cache for the comb. kernel
gf_('send_command', 'set_kernel COMBINED 200');

%other kernels (linear 10 mb cache, gaussian 20 mb cache sigma 1, poly 50mb
%cache degreee 3 homogene
gf_('send_command', 'add_kernel 1 LINEAR REAL 10 ');
gf_('send_command', 'add_kernel 2 GAUSSIAN REAL 20 1');
gf_('send_command', 'add_kernel 3 POLY REAL 50 3 0');

gf_('send_command', 'init_kernel TRAIN');
gf_('send_command', 'c 5');
gf_('send_command', 'new_svm LIGHT');
gf_('send_command', 'svm_train');


gf_('send_command', 'init_kernel TEST');
Ks=gf_('get_kernel_matrix') ;

gf_('set_features', 'TRAIN', XT1);
gf_('set_features', 'TEST', XT1);
gf_('send_command', 'set_kernel LINEAR REAL 10 ');
gf_('send_command', 'init_kernel TRAIN');
gf_('send_command', 'init_kernel TEST');
K1=gf_('get_kernel_matrix') ;


gf_('set_features', 'TRAIN', XT2);
gf_('set_features', 'TEST', XT2);
gf_('send_command', 'set_kernel GAUSSIAN REAL 20 1');
gf_('send_command', 'init_kernel TRAIN');
gf_('send_command', 'init_kernel TEST');
K2=gf_('get_kernel_matrix') ;



gf_('set_features', 'TRAIN', XT3);
gf_('set_features', 'TEST', XT3);
gf_('send_command', 'set_kernel POLY REAL 50 3 0');
gf_('send_command', 'init_kernel TRAIN');
gf_('send_command', 'init_kernel TEST');
K3=gf_('get_kernel_matrix') ;


norm(Ks-1*K1-2*K2-3*K3)
