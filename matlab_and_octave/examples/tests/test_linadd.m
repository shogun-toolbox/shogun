C=10;
svm_eps=1e-5;
debug=0;

num=60000;
dims=1000;
numval=10000;

rand('state',sum(100*clock));
traindat=[ (rand(dims,num/2)-0.1) (rand(dims,num/2)+0.1) ];
trainlab=[ -ones(1,num/2) ones(1,num/2) ];
valdat=[ (rand(dims,numval/2)-0.1) (rand(dims,numval/2)+0.1) ];
vallab=[ -ones(1,numval/2) ones(1,numval/2) ];

sg('send_command', 'loglevel ALL');

sg('send_command', 'use_mkl 0') ;
sg('send_command', 'use_linadd 1') ;
sg('send_command', 'use_precompute 0') ;
sg('send_command', 'mkl_parameters 1e-5 0') ;
sg('send_command', sprintf('svm_epsilon %f', svm_eps)) ;
sg('send_command', 'clean_features TRAIN') ;
sg('send_command', 'clean_kernel') ;

sg('set_features', 'TRAIN', traindat);
sg('set_labels', 'TRAIN', trainlab);
sg('send_command', 'set_kernel LINEAR REAL 10 1.0'); %die 1.0 entspricht scaling
sg('send_command', 'init_kernel TRAIN');
sg('send_command', 'new_svm LIGHT');
sg('send_command', sprintf('c %f', C));
sg('send_command', 'svm_train');
[b, alpha_tmp]=sg('get_svm');
sg('send_command', 'init_kernel_optimization') ;

sg('set_features', 'TEST', valdat);
sg('set_labels', 'TEST', vallab);
sg('send_command', 'init_kernel TEST');
sg('send_command', 'init_kernel_optimization');
w=sg('get_kernel_optimization');


out=sg('svm_classify');
valerr=mean(vallab~=sign(out));
valerr

if debug == 1
	alphas=zeros(1,length(trainlab));
	alphas(alpha_tmp(:,2)+1)=alpha_tmp(:,1);

	idx=find(alphas);
	w=0;
	for i=1:length(idx),
		w=w+alphas(idx(i))*traindat(:,idx(i));
	end

	abs(w'*valdat(:,1)+b - out(1)) % sollte <1e-9 oder so sein

	w2=sg('get_kernel_optimization');
	max(abs(w-w2))
	abs(w'*valdat(:,1)+b - out(1)) % sollte <1e-9 oder so sein
end
