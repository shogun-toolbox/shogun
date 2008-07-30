function y = preproc(filename)
	addpath('util');
	addpath('../data/preproc');
	y=true;

	eval('globals'); % ugly hack to have vars from filename as globals
	eval(filename);

	if ~set_features()
		return;
	end

	pname=fix_preproc_name_inconsistency(name);
	if strcmp(pname, 'PRUNEVARSUBMEAN')==1
		sg('add_preproc', pname, tobool(preproc_arg0_divide));
	else
		sg('add_preproc', pname);
	end

	sg('attach_preproc', 'TRAIN');
	sg('attach_preproc', 'TEST');

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
