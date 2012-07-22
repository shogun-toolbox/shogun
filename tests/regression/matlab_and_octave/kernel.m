function y = kernel(filename)
	addpath('util');
	addpath('../data/kernel');
	y=true;

	eval('globals'); % ugly hack to have vars from filename as globals
	%system(sprintf('ln -sf ../data/kernel/%s.m testscript.m', filename));
	%testscript;
	%system('rm -f testscript.m'); %avoid ultra long filenames (>63 chars)
	eval(filename);

	if ~set_features('kernel_')
		return;
	end

	if ~set_kernel()
		return;
	end

	kmatrix=sg('get_kernel_matrix', 'TRAIN');
	km_train=max(max(abs(kernel_matrix_train-kmatrix)));

	kmatrix=sg('get_kernel_matrix', 'TEST');
	km_test=max(max(abs(kernel_matrix_test-kmatrix)));

	data={'kernel', km_train, km_test};
	y=check_accuracy(kernel_accuracy, data);
