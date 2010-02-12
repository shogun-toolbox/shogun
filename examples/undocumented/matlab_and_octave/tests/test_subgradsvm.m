C=1;
epsilon=1e-3;

run_subgradientsvm=1;
run_libsvm=0;
run_svmlight=1;
run_gpbtsvm=1;

dataset=7;

switch dataset
case 1,
	load /home/sonne/vojtech/subgradsvm/uci_spambase.mat
	data=[];
	traindat=x';
	trainlab=t';
	testdat=x';
	testlab=t';
case 2,
	load ~/subgradient/data/covertype.mat
	for i=1:size(x,2),
		x(:,i)=x(:,i)/norm(x(:,i));
	end
	traindat=x;
	trainlab=y';
	testdat=x;
	testlab=y';
	%addpath ../util
	%write_svmlight_format(x,y, '../data/covertype_norm_one.svmlight');
	%keyboard
case 3,
	load ../data/mnist8n8-10k.mat

	for i=1:size(x,2),
		x(:,i)=x(:,i)/norm(x(:,i));
	end
	traindat=full(x);
	trainlab=full(y);
	testdat=full(x);
	testlab=full(y);

case 4,
	load ~/subgradient/data/uciadu6.mat
	traindat=full(x);
	trainlab=full(y);
	testdat=full(x);
	testlab=full(y);
case 5,
	load ~/subgradient/data/web6a.mat
	traindat=full(x);
	trainlab=full(y);
	testdat=full(x);
	testlab=full(y);
case 6,
	rand('state',17);
	num=20000;
	dim=100;
	dist=0.03;
	%num=20;
	%dim=1000;
	%dist=0.01;

	traindat=[rand(dim,num/2)-dist, rand(dim,num/2)+dist];
	traindat=traindat/(dim*mean(traindat(:)));
	trainlab=[-ones(1,num/2), ones(1,num/2) ];

	testdat=[rand(dim,num/2)-dist, rand(dim,num/2)+dist];
	testdat=testdat/(dim*mean(testdat(:)));;
	testlab=[-ones(1,num/2), ones(1,num/2) ];
case 7,
	load ~/astro-ph_29882.mat
	%for i=1:size(x,2)
	%	x(i,:)=x(i,:)-mean(x(i,:));
	%end
	%for i=1:size(x,2)
	%	x(i,:)=x(i,:)/std(x(i,:));
	%end
	%keyboard
	traindat=x;
	trainlab=y;
otherwise
	error('unknown dataset')
end

%sg('loglevel', 'ALL');

%%%%SUBGRADIENT%%%
if run_subgradientsvm,
	sg('set_features', 'TRAIN', traindat);

	sg('set_labels', 'TRAIN', trainlab);
	sg('c', C);
	sg('svm_epsilon', epsilon);
	%sg('new_classifier', 'SVMLIN');
	sg('svm_qpsize', 100);
	sg('svm_max_qpsize', 1000);
	sg('new_classifier', 'SUBGRADIENTSVM');
	tic;
	sg('train_classifier');
	timesubgradsvm=toc

	sg('set_features', 'TEST', traindat);
	trainout=sg('classify');
	trainerr=mean(trainlab~=sign(trainout))
	[b,W]=sg('get_classifier');
	F_subgrad = 0.5*norm(W)^2 + C*sum(max(zeros(size(trainout)),1 - trainlab.*trainout))
end

%%%%LIGHT%%%
if run_svmlight,
	sg('set_features', 'TRAIN', traindat);
	sg('set_labels', 'TRAIN', trainlab);
	sg('c', C);
	sg('set_kernel', 'LINEAR', 'SPARSEREAL', 10, 1.0);
	sg('svm_qpsize', 42);
	sg('svm_epsilon', epsilon);
	sg('new_classifier', 'SVMLIGHT');
	tic;
	sg('train_classifier');
	timelight=toc

	sg('init_kernel_optimization');
	sg('set_features', 'TEST', traindat);
	obj_light=sg('get_svm_objective')
	trainout_reflight=sg('classify');
	trainerr_reflight=mean(trainlab~=sign(trainout_reflight))

	W=sg('get_kernel_optimization');
	F_light = 0.5*norm(W)^2 + C*sum(max(zeros(size(trainout_reflight)),1 - trainlab.*trainout_reflight));
	F_light
end

%%%%%GPBT%%%
if run_gpbtsvm,
	sg('set_features', 'TRAIN', traindat);
	sg('set_labels', 'TRAIN', trainlab);
	sg('c', C);
	sg('set_kernel', 'LINEAR', 'SPARSEREAL', 10, 1.0);
	sg('svm_qpsize', 500);
	sg('svm_epsilon', epsilon);
	sg('new_classifier', 'GPBTSVM');
	tic;
	sg('train_classifier');
	timegpbt=toc

	sg('init_kernel_optimization');
	sg('set_features', 'TEST', traindat);
	obj_gpbt=sg('get_svm_objective')
	trainout_refgpbt=sg('classify');
	trainerr_refgpbt=mean(trainlab~=sign(trainout_refgpbt))

	W=sg('get_kernel_optimization');
	F_gpbt = 0.5*norm(W)^2 + C*sum(max(zeros(size(trainout_refgpbt)),1 - trainlab.*trainout_refgpbt));
	F_gpbt
end


%%%%%LIBSVM%%%
if run_libsvm,
	sg('set_features', 'TRAIN', traindat);
	sg('set_labels', 'TRAIN', trainlab);
	sg('c', C);
	sg('set_kernel', 'LINEAR', 'SPARSEREAL', 1000, 1.0);
	sg('svm_epsilon', epsilon);
	sg('new_classifier', 'LIBSVM');
	tic;
	sg('train_classifier');
	timelibsvm=toc

	sg('init_kernel_optimization');
	sg('set_features', 'TEST', traindat);
	obj_libsvm=sg('get_svm_objective')
	trainout_reflibsvm=sg('classify');
	trainerr_reflibsvm=mean(trainlab~=sign(trainout_reflibsvm))

	W=sg('get_kernel_optimization');
	F_libsvm = 0.5*norm(W)^2 + C*sum(max(zeros(size(trainout_reflibsvm)),1 - trainlab.*trainout_reflibsvm));
	F_libsvm
	%
	%[b,a]=sg('get_classifier');
	%alpha=zeros(size(traindat,2),1);
	%alpha(a(:,2)+1)=a(:,1);

	%F_libsvm_alpha=0.5*alpha'*(traindat'*traindat)*alpha + C*sum(max(zeros(size(trainout_reflibsvm)),1 - trainout_reflibsvm))
end

disp('training times')
if run_svmlight,
	fprintf('light:%f\n',timelight)
end
if run_libsvm,
	fprintf('libsvm:%f\n',timelibsvm)
end
if run_subgradientsvm,
	fprintf('subgrad:%f\n',timesubgradsvm)
end
if run_gpbtsvm,
	fprintf('gpbt:%f\n',timegpbt)
end

disp('objectives')
if run_svmlight,
	fprintf('light:%f\n',F_light)
end
if run_libsvm,
	fprintf('libsvm:%f\n',F_libsvm)
end
if run_subgradientsvm,
	fprintf('subgrad:%f\n',F_subgrad)
end
if run_gpbtsvm,
	fprintf('gpbt:%f\n',F_gpbt)
end

disp('trainerr')
if run_svmlight,
	fprintf('light:%f\n',trainerr_reflight)
end
if run_libsvm,
	fprintf('libsvm:%f\n',trainerr_reflibsvm)
end
if run_subgradientsvm,
	fprintf('subgrad:%f\n',trainerr)
end
if run_gpbtsvm,
	fprintf('gpbt:%f\n',trainerr_refgpbt)
end

size(x)
size(y)
