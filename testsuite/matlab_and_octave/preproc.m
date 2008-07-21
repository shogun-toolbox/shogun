function y = preproc(filename)
	addpath('util');
	addpath('../data/preproc');

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

	pname=fix_preproc_name_inconsistency(name);
	if strcmp(pname, 'PRUNEVARSUBMEAN')==1
		sg('add_preproc', pname, tobool(preproc_arg0_divide));
	else
		sg('add_preproc', pname);
	end

	sg('attach_preproc', 'TRAIN');
	sg('attach_preproc', 'TEST');

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
