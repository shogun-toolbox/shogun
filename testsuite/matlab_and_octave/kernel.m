function y = kernel(filename)
	addpath('util');
	addpath('../data/kernel');
	y=true;

	eval('globals'); % ugly hack to have vars from filename as globals
	eval(filename);

	if !set_features()
		return;
	end

	if !set_kernel()
		return;
	end

	kmatrix=sg('get_kernel_matrix');
	ktrain=max(abs(km_train-kmatrix))(1:1);

	sg('init_kernel', 'TEST');
	kmatrix=sg('get_kernel_matrix');
	ktest=max(abs(km_test-kmatrix))(1:1);

	data={'kernel', ktrain, ktest};
	y=check_accuracy(accuracy, data);
