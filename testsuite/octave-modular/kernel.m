function y = kernel(filename)
	addpath('util');
	addpath('../data/kernel');

	eval('globals'); % ugly hack to have vars from filename as globals
	eval(filename);

	fset=set_features();
	if !fset
		y=false;
		return;
	elseif strcmp(fset, 'catchme')==1
		y=true;
		return;
	end

	kset=set_and_train_kernel();
	if !kset
		y=false;
		return;
	elseif strcmp(kset, 'catchme')==1
		y=true;
		return;
	end

	kmatrix=sg('get_kernel_matrix');
	ktrain=max(abs(km_train-kmatrix))(1:1);

	sg('init_kernel', 'TEST');
	kmatrix=sg('get_kernel_matrix');
	ktest=max(abs(km_test-kmatrix))(1:1);

	y=check_accuracy(accuracy, ktrain, ktest);
