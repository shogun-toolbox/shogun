cache_size=50;
C=0.000001;
numtrain=1000;
svm_eps=1e-4;
mkl_eps=1e-4;
W0=0.1;
W1=0.5;
W2=2;
W3=1;

rand('state',0);
traindat=rand(120,numtrain);
trainlab=sign(rand(1,numtrain)-0.5);
testdat=traindat;
testlab=trainlab;

kmtrain{1}=ones(numtrain,numtrain);
kmtrain{2}=ones(numtrain,numtrain);
kmtrain{3}=ones(numtrain,numtrain);
kmtrain{4}=ones(numtrain,numtrain);
kmtrain{5}=eye(numtrain,numtrain);

kmtest{1}=ones(numtrain,numtrain);
kmtest{2}=ones(numtrain,numtrain);
kmtest{3}=ones(numtrain,numtrain);
kmtest{4}=ones(numtrain,numtrain);
kmtest{5}=eye(numtrain,numtrain);

gf('send_command', 'new_svm LIGHT');
gf('send_command', 'clean_features TRAIN');
gf('send_command', 'clean_kernels') ;

gf('set_labels', 'TRAIN', trainlab);
gf('set_features','TRAIN', traindat);
gf('send_command', sprintf('set_kernel GAUSSIAN REAL %d %f', cache_size, W0));
gf('send_command', 'init_kernel TRAIN');
kmcool=gf('get_kernel_matrix');

gf('send_command', 'clean_features TRAIN');
gf('send_command', 'clean_kernels') ;
gf('add_features','TRAIN', traindat);
gf('add_features','TRAIN', traindat);
gf('add_features','TRAIN', traindat);
gf('add_features','TRAIN', traindat);
gf('add_features','TRAIN', traindat);
gf('add_features','TRAIN', traindat);
gf('add_features','TRAIN', traindat);
gf('set_labels', 'TRAIN', trainlab);
gf('send_command', sprintf('set_kernel COMBINED %d', cache_size));
gf('send_command', sprintf('add_kernel 1 GAUSSIAN REAL %d %f', cache_size, W1));
gf('send_command', sprintf('add_kernel 1 GAUSSIAN REAL %d %f', cache_size, W2));
gf('send_command', sprintf('add_kernel 1 GAUSSIAN REAL %d %f', cache_size, W3));
gf('send_command', sprintf('add_kernel 1 CUSTOM ANY %d', cache_size));
gf('set_custom_kernel',kmtrain{1},'FULL');
gf('send_command', sprintf('add_kernel 1 CUSTOM ANY %d', cache_size));
gf('set_custom_kernel',kmtrain{2},'FULL');
gf('send_command', sprintf('add_kernel 1 CUSTOM ANY %d', cache_size));
gf('set_custom_kernel',kmtrain{3},'FULL');
gf('send_command', sprintf('add_kernel 1 CUSTOM ANY %d', cache_size));
gf('set_custom_kernel',kmcool,'FULL');

gf('send_command', 'init_kernel TRAIN');
gf('send_command', 'use_mkl 1');
gf('send_command', 'use_precompute 0');
gf('send_command', sprintf('mkl_parameters %f 0', mkl_eps));
gf('send_command', sprintf('c %f',C));
gf('send_command', sprintf('svm_epsilon %f',svm_eps));
gf('send_command', 'svm_train');
[b, alphas]=gf('get_svm');
ws=gf('get_subkernel_weights');

gf('send_command', 'clean_features TEST');
gf('send_command', 'clean_kernels') ;
gf('send_command', sprintf('set_kernel COMBINED %d', cache_size));
gf('send_command', sprintf('add_kernel 1 GAUSSIAN REAL %d %f', cache_size, W1));
gf('send_command', sprintf('add_kernel 1 GAUSSIAN REAL %d %f', cache_size, W2));
gf('send_command', sprintf('add_kernel 1 GAUSSIAN REAL %d %f', cache_size, W3));
gf('send_command', sprintf('add_kernel 1 CUSTOM ANY %d', cache_size));
gf('set_custom_kernel',kmtest{1},'FULL');
gf('send_command', sprintf('add_kernel 1 CUSTOM ANY %d', cache_size));
gf('set_custom_kernel',kmtest{2},'FULL');
gf('send_command', sprintf('add_kernel 1 CUSTOM ANY %d', cache_size));
gf('set_custom_kernel',kmtest{3},'FULL');
gf('send_command', sprintf('add_kernel 1 CUSTOM ANY %d', cache_size));
gf('set_custom_kernel',kmcool,'FULL');
gf('set_subkernel_weights',ws);
gf('add_features','TEST', testdat);
gf('add_features','TEST', testdat);
gf('add_features','TEST', testdat);
gf('add_features','TEST', testdat);
gf('add_features','TEST', testdat);
gf('add_features','TEST', testdat);
gf('add_features','TEST', testdat);
gf('set_labels', 'TEST', testlab);
gf('send_command', 'init_kernel TEST');
out=gf('svm_classify');
mean(sign(out)~=testlab)
