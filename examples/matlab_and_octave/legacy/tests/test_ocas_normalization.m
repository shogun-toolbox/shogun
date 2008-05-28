acgt='ACGT';
rand('state',17')


%load /home/sonne/Documents/work/first/teach/bioinfapps/data/splice_data.mat
%traindat=data(70:130,1:1000);
%trainlab=label(1:1000);
traindat=acgt(ceil(4*rand(100,100)));
trainlab=[-ones(1,50), ones(1,50)];

C=1;
order=6;
from_order=40;
max_mismatch=0;
cache=100;
normalize=1;
mkl_stepsize=1;
block=1;
single_degree=-1;
epsilon=1e-6;

testdat=traindat;
testlab=trainlab;

%train svm
sg('send_command', 'threads 1');
sg('send_command', 'use_linadd 1');
sg('send_command', 'use_batch_computation 1');
sg('send_command', 'loglevel ALL');
sg('set_features', 'TRAIN', traindat, 'DNA');
sg('set_labels', 'TRAIN', trainlab);
sg('send_command', 'svm_use_bias 0');
sg('send_command', sprintf('svm_epsilon %f', epsilon));
sg('send_command', 'new_svm LIGHT');

sg('send_command', sprintf( 'set_kernel WEIGHTEDDEGREE CHAR %i %i %i %i %i %i %i', cache, from_order, max_mismatch, normalize, mkl_stepsize, block, single_degree) );
sg('send_command', 'init_kernel TRAIN');
x=sg('get_subkernel_weights');

sg('send_command', sprintf( 'set_kernel WEIGHTEDDEGREE CHAR %i %i %i %i %i %i %i', cache, order, max_mismatch, 0, mkl_stepsize, block, single_degree) );
sg('send_command', 'init_kernel TRAIN');
sg('set_subkernel_weights',x(1:order));
kmu=sg('get_kernel_matrix');

sg('send_command', sprintf( 'set_kernel WEIGHTEDDEGREE CHAR %i %i %i %i %i %i %i', cache, order, max_mismatch, normalize, mkl_stepsize, block, single_degree) );
sg('send_command', 'init_kernel TRAIN');
sg('set_subkernel_weights',x(1:order));
km=sg('get_kernel_matrix');

sg('send_command', 'new_svm LIGHT');
sg('send_command', sprintf('c %f',C));
tic;
% this is only necessary for svm_train not to choke on positive definites
sg('send_command', 'svm_train');
tim_lo=toc;

%evaluate svm on test data
sg('set_features', 'TEST', testdat, 'DNA');
sg('send_command', 'init_kernel TEST');
out_ref=sg('svm_classify');
prc_ref=calcrfcscore(out_ref, testlab);
roc_ref=calcrocscore(out_ref, testlab);


traindat(traindat=='A')=0;
traindat(traindat=='C')=1;
traindat(traindat=='G')=2;
traindat(traindat=='T')=3;
traindat=uint8(traindat);

testdat(testdat=='A')=0;
testdat(testdat=='C')=1;
testdat(testdat=='G')=2;
testdat(testdat=='T')=3;
testdat=uint8(testdat);

%epsilon=1e-3;
epsilon=1e-6;
sg('send_command', 'loglevel ALL');
sg('set_features', 'TRAIN', traindat, 'RAWDNA');
sg('set_labels', 'TRAIN', trainlab);
sg('send_command', sprintf('c %f',C));
sg('send_command', sprintf('svm_epsilon %f', epsilon));
sg('send_command', sprintf('new_svm WDSVMOCAS %d %d', order, from_order));
tic;
sg('send_command', 'svm_train');
tim_lo=toc;

%evaluate svm on test data
sg('set_features', 'TEST', testdat, 'RAWDNA');
out=sg('svm_classify');
prc=calcrfcscore(out, testlab);
roc=calcrocscore(out, testlab);

max(abs(out_ref-out))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%dat=[];
%weights=sqrt((order:-1:1)/sum(order:-1:1));
%
%N = size(traindat,1);
%nDim = 0;
%for d = 1:order,
%	nDim = nDim + 4^d;
%end
%nDim = nDim*N;
%
%for j=1:size(traindat,2),
%	dat(:,j)= zeros(nDim,1);
%	offset = 0;
%	for i=1:N,
%	   val = 0;
%	   for d = 1:order
%		   if i+d-1<=N,
%			 val = 4*val + double(traindat(i+d-1,j));
%			 dat(offset+val+1,j) = weights(d);
%			 offset = offset + 4^d;
%		   end
%	   end    
%	end
%end
%
%
%traindat=sparse(dat)/sqrt(kmu(1));
%testdat=traindat;
%
%sg('send_command', 'loglevel ALL');
%sg('set_features', 'TRAIN', traindat);
%sg('set_labels', 'TRAIN', trainlab);
%sg('send_command', sprintf('c %f', C));
%sg('send_command', sprintf('svm_epsilon %f', epsilon));
%sg('send_command', 'svm_use_bias 0');
%sg('send_command', 'new_classifier SVMOCAS');
%sg('send_command', 'train_classifier');
%sg('set_features', 'TEST', testdat);
%out_ocas=sg('classify');
%max(abs(out_ref-out))
%max(abs(out_ocas-out_ref))
%max(abs(out_ocas-out))
