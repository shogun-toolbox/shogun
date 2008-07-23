function y = preproc(filename)
	init_shogun

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

	if strcmp(name, 'LogPlusOne')==1
		preproc=LogPlusOne();
	elseif strcmp(name, 'NormOne')==1
		preproc=NormOne();
	elseif strcmp(name, 'PruneVarSubMean')==1
		preproc=PruneVarSubMean(tobool(preproc_arg0_divide));
	elseif strcmp(name, 'SortUlongString')==1
		preproc=SortUlongString();
	elseif strcmp(name, 'SortWord')==1
		preproc=SortWord();
	elseif strcmp(name, 'SortWordString')==1
		preproc=SortWordString();
	end

	preproc.init(feats_train);
	feats_train.add_preproc(preproc);
	feats_train.apply_preproc();
	feats_test.add_preproc(preproc);
	feats_test.apply_preproc();

	kset=set_kernel();
	if !kset
		y=false;
		return;
	elseif strcmp(kset, 'catchme')==1
		y=true;
		return;
	end

	kmatrix=kernel.get_kernel_matrix();
	ktrain=max(abs(km_train-kmatrix))(1:1);

	kernel.init(feats_train, feats_test);
	kmatrix=kernel.get_kernel_matrix();
	ktest=max(abs(km_test-kmatrix))(1:1);

	y=check_accuracy(accuracy, ktrain, ktest);
