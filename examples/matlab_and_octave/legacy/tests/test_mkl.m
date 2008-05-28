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

sg('send_command', 'new_svm LIGHT');
sg('send_command', 'clean_features TRAIN');
sg('send_command', 'clean_kernel') ;

sg('set_labels', 'TRAIN', trainlab);
sg('set_features','TRAIN', traindat);
sg('send_command', sprintf('set_kernel GAUSSIAN REAL %d %f', cache_size, W0));
sg('send_command', 'init_kernel TRAIN');
kmcool=sg('get_kernel_matrix');

sg('send_command', 'clean_features TRAIN');
sg('send_command', 'clean_kernel') ;
sg('add_features','TRAIN', traindat);
sg('add_features','TRAIN', traindat);
sg('add_features','TRAIN', traindat);
sg('add_features','TRAIN', traindat);
sg('add_features','TRAIN', traindat);
sg('add_features','TRAIN', traindat);
sg('add_features','TRAIN', traindat);
sg('set_labels', 'TRAIN', trainlab);
sg('send_command', sprintf('set_kernel COMBINED %d', cache_size));
sg('send_command', sprintf('add_kernel 1 GAUSSIAN REAL %d %f', cache_size, W1));
sg('send_command', sprintf('add_kernel 1 GAUSSIAN REAL %d %f', cache_size, W2));
sg('send_command', sprintf('add_kernel 1 GAUSSIAN REAL %d %f', cache_size, W3));
sg('send_command', sprintf('add_kernel 1 CUSTOM ANY %d', cache_size));
sg('set_custom_kernel',kmtrain{1},'FULL');
sg('send_command', sprintf('add_kernel 1 CUSTOM ANY %d', cache_size));
sg('set_custom_kernel',kmtrain{2},'FULL');
sg('send_command', sprintf('add_kernel 1 CUSTOM ANY %d', cache_size));
sg('set_custom_kernel',kmtrain{3},'FULL');
sg('send_command', sprintf('add_kernel 1 CUSTOM ANY %d', cache_size));
sg('set_custom_kernel',kmcool,'FULL');

sg('send_command', 'init_kernel TRAIN');
sg('send_command', 'use_mkl 1');
sg('send_command', 'use_precompute 0');
sg('send_command', sprintf('mkl_parameters %f 0', mkl_eps));
sg('send_command', sprintf('c %f',C));
sg('send_command', sprintf('svm_epsilon %f',svm_eps));
sg('send_command', 'svm_train');
[b, alphas]=sg('get_svm');
ws=sg('get_subkernel_weights');

sg('send_command', 'clean_features TEST');
sg('send_command', 'clean_kernel') ;
sg('send_command', sprintf('set_kernel COMBINED %d', cache_size));
sg('send_command', sprintf('add_kernel 1 GAUSSIAN REAL %d %f', cache_size, W1));
sg('send_command', sprintf('add_kernel 1 GAUSSIAN REAL %d %f', cache_size, W2));
sg('send_command', sprintf('add_kernel 1 GAUSSIAN REAL %d %f', cache_size, W3));
sg('send_command', sprintf('add_kernel 1 CUSTOM ANY %d', cache_size));
sg('set_custom_kernel',kmtest{1},'FULL');
sg('send_command', sprintf('add_kernel 1 CUSTOM ANY %d', cache_size));
sg('set_custom_kernel',kmtest{2},'FULL');
sg('send_command', sprintf('add_kernel 1 CUSTOM ANY %d', cache_size));
sg('set_custom_kernel',kmtest{3},'FULL');
sg('send_command', sprintf('add_kernel 1 CUSTOM ANY %d', cache_size));
sg('set_custom_kernel',kmcool,'FULL');
sg('set_subkernel_weights',ws);
sg('add_features','TEST', testdat);
sg('add_features','TEST', testdat);
sg('add_features','TEST', testdat);
sg('add_features','TEST', testdat);
sg('add_features','TEST', testdat);
sg('add_features','TEST', testdat);
sg('add_features','TEST', testdat);
sg('set_labels', 'TEST', testlab);
sg('send_command', 'init_kernel TEST');
out=sg('svm_classify');
mean(sign(out)~=testlab)
