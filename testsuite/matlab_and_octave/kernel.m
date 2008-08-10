function y = kernel(filename)
	addpath('util');
	addpath('../data/kernel');
	y=true;

	eval('globals'); % ugly hack to have vars from filename as globals
	system(sprintf('ln -sf ../data/kernel/%s.m testscript.m', filename));
	testscript;
	system('rm -f testscript.m'); %avoid ultra long filenames (>63 chars)
	%eval(filename);

	if ~set_features()
		return;
	end

	if ~set_kernel()
		return;
	end

	kmatrix=sg('get_kernel_matrix');
	ktrain=max(max(abs(km_train-kmatrix)));

	sg('init_kernel', 'TEST');
	kmatrix=sg('get_kernel_matrix');
	ktest=max(max(abs(km_test-kmatrix)));

	data={'kernel', ktrain, ktest};
	y=check_accuracy(accuracy, data);
