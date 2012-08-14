function y = preprocessor(filename)
	addpath('util');
	addpath('../data/preprocessor');
	y=true;

	eval('globals'); % ugly hack to have vars from filename as globals
	eval(filename);

	if ~set_features('kernel_')
		return;
	end

	pname=fix_preproc_name_inconsistency(preprocessor_name);
	if strcmp(pname, 'PRUNEVARSUBMEAN')==1
		sg('add_preproc', pname, tobool(preprocessor_arg0_divide));
	else
		sg('add_preproc', pname);
	end

	sg('attach_preproc', 'TRAIN');
	sg('attach_preproc', 'TEST');

	if ~set_kernel()
		return;
	end

	kmatrix=sg('get_kernel_matrix', 'TRAIN');
	km_train=max(max(abs(kernel_matrix_train-kmatrix)));

	kmatrix=sg('get_kernel_matrix', 'TEST');
	km_test=max(max(abs(kernel_matrix_test-kmatrix)));

	data={'kernel', km_train, km_test};
	y=check_accuracy(kernel_accuracy, data);
