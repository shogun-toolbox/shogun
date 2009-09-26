cache_size=50;
C=1;
svm_eps=1e-5;
mkl_eps=1e-6;
svm_tube=0.01;

kernels={};

global lpenv ;
if isempty(lpenv)|lpenv==0,
 lpenv=cplex_license(0) ;
end ;

% Kernel width for the 5 "basic" SVMs
rbf_width(1) = 0.005;
rbf_width(2) = 0.05;
rbf_width(3) = 0.5;
rbf_width(4) = 1;
rbf_width(5) = 10;
rand('state',0);
%rand('state',sum(100*clock));
f = [0.1:0.2:5];   % values for the different frequencies
no_obs = 1000;     % number of observations
kk=4;
traindat = [1:(((10*2*pi)-1)/(no_obs-1)):10*2*pi];
trainlab = sin(f(kk)*traindat);
testdat = [1:(((10*2*pi)-1)/(no_obs-1)):10*2*pi];
testlab = sin(f(kk)*traindat);

sg('new_regression', 'LIBSVR');
%sg('new_regression', 'SVRLIGHT');
sg('clean_features', 'TRAIN');
sg('set_features','TRAIN', traindat);
sg('set_labels', 'TRAIN', trainlab);
sg('set_kernel', 'GAUSSIAN', 'REAL', cache_size, rbf_width(1));
kernels{1}=sg('get_kernel_matrix', 'TRAIN');

sg('set_features','TRAIN', traindat);
sg('set_kernel', 'GAUSSIAN', 'REAL', cache_size, rbf_width(2));
kernels{2}=sg('get_kernel_matrix', 'TRAIN');

sg('set_features','TRAIN', traindat);
sg('set_kernel', 'GAUSSIAN', 'REAL', cache_size, rbf_width(3));
kernels{3}=sg('get_kernel_matrix', 'TRAIN');

sg('set_features','TRAIN', traindat);
sg('set_kernel', 'GAUSSIAN', 'REAL', cache_size, rbf_width(4));
kernels{4}=sg('get_kernel_matrix', 'TRAIN');

sg('set_features','TRAIN', traindat);
sg('set_kernel', 'GAUSSIAN', 'REAL', cache_size, rbf_width(5));
kernels{5}=sg('get_kernel_matrix', 'TRAIN');

sg('clean_features',  'TRAIN');
sg('add_features','TRAIN', traindat);
sg('add_features','TRAIN', traindat);
sg('add_features','TRAIN', traindat);
sg('add_features','TRAIN', traindat);
sg('add_features','TRAIN', traindat);
sg('set_kernel', 'COMBINED', cache_size);
sg('add_kernel', 1, 'GAUSSIAN', 'REAL', cache_size, rbf_width(1));
sg('add_kernel', 1, 'GAUSSIAN', 'REAL', cache_size, rbf_width(2));
sg('add_kernel', 1, 'GAUSSIAN', 'REAL', cache_size, rbf_width(3));
sg('add_kernel', 1, 'GAUSSIAN', 'REAL', cache_size, rbf_width(4));
sg('add_kernel', 1, 'GAUSSIAN', 'REAL', cache_size, rbf_width(5));

sg('use_mkl', 0);
sg('loglevel', 'ALL');
sg('mkl_parameters', 1e-3, 0);
sg('c', C);
sg('svm_epsilon', svm_eps);
sg('svr_tube_epsilon', svm_tube);

betas=sg('get_subkernel_weights') ;
betas=betas/sum(betas(:)) ;
sg('set_subkernel_weights',betas) ;

sg('c', C);


OBJ=[] ;
thetas=[] ; sumbetas=[] ; A=[] ;

alpha_svmlight={};
b_svmlight=[];
beta_svmlight={};
obj_svmlight=[];
alpha_libsvm={};
b_libsvm=[];
beta_libsvm={};
obj_libsvm=[];

tic
for ii=1:100,
	% find most violated constraints
	% 1. compute optimal alphas
	sg('new_regression', 'LIBSVR');
	%sg('new_regression', 'SVRLIGHT');
	sg('train_regression');
	betas=sg('get_subkernel_weights') ;
	[b,alpha_idx]=sg('get_svm') ;
	alpha_svmlight{ii}=alpha_idx;
	b_svmlight(ii)=b;
	beta_svmlight{ii}=betas;
	obj_svmlight(ii)=sg('get_svm_objective');


	%sg('new_regression', 'LIBSVR');
	sg('new_regression', 'SVRLIGHT');
	sg('train_regression');
	betas=sg('get_subkernel_weights') ;
	[b,alpha_idx]=sg('get_svm') ;
	alpha_libsvm{ii}=alpha_idx;
	b_libsvm(ii)=b;
	beta_libsvm{ii}=betas;
	obj_libsvm(ii)=sg('get_svm_objective');

	obj_libsvm(ii)-obj_svmlight(ii)
	if abs(obj_libsvm(ii)-obj_svmlight(ii))>1e-3 | abs(b_libsvm(ii)-b_svmlight(ii))>1e-3,
		obj_libsvm(ii)
		obj_svmlight(ii)
		b_libsvm(ii)-b_svmlight(ii)
		b_libsvm(ii)
		b_svmlight(ii)
		keyboard
	end

	alphas=zeros(1,size(traindat,2)) ;
	alphas(alpha_idx(:,2)+1)=alpha_idx(:,1) ;

	% 2. compute current SVM objective
	OBJ(ii)=sg('get_svm_objective');

	sum_contrib=0;
	for i=1:5,
		contrib(i)=0.5*alphas*kernels{i}*alphas';
		sum_contrib=sum_contrib+contrib(i);
	end

	sum(abs(alphas))*svm_tube-alphas*trainlab'+sum_contrib
	const_contrib=sum(abs(alphas))*svm_tube-alphas*trainlab';
	for i=1:5,
		sumbetas(i)=const_contrib+contrib(i);
	end

	INF=1e20 ;
	A=[A; -sumbetas -1] ;
	f=[zeros(1,length(sumbetas)) 1] ;
	CA=[ones(1,length(sumbetas)) 0
		A] ;
	b=[1;zeros(size(A,1),1)] ;
	lb=[zeros(1,length(sumbetas)) -INF] ;
	ub=[ones(1,length(sumbetas)) INF] ;
	[res,lambda,how]=lp_solve(lpenv, f', sparse(CA), b, lb',ub',1,0) ;
	betas=reshape(res(1:end-1)',size(betas,1),size(betas,2)) ;
	thetas(ii)=-res(end) ;

	disp(sprintf('relative gap: %f',abs(1-thetas(end)/OBJ(end)))) ;
	betas
	if abs(1-thetas(end)/OBJ(end))<mkl_eps
		break
	end

	% update betas in gf
	sg('set_subkernel_weights',betas) ;
end
toc

sg('clean_features', 'TEST' );
sg('add_features','TEST', testdat);
sg('add_features','TEST', testdat);
sg('add_features','TEST', testdat);
sg('add_features','TEST', testdat);
sg('add_features','TEST', testdat);
sg('set_labels', 'TEST', testlab);
out2=sg('classify');

sum(abs(testlab-out2))
sum((testlab-out2).^2)
