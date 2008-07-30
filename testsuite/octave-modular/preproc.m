function y = preproc(filename)
	init_shogun;
	y=true;

	addpath('util');
	addpath('../data/preproc');

	eval('globals'); % ugly hack to have vars from filename as globals
	eval(filename);

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
	else
		error('Unsupported preproc %s', name);
	end

	if ~set_features()
		return;
	end

	preproc.init(feats_train);
	feats_train.add_preproc(preproc);
	feats_train.apply_preproc();
	feats_test.add_preproc(preproc);
	feats_test.apply_preproc();

	if ~set_kernel()
		return;
	end

	ktrain=max(max(abs(km_train-kernel.get_kernel_matrix())));
	kernel.init(feats_train, feats_test);
	ktest=max(max(abs(km_test-kernel.get_kernel_matrix())));

	data={'kernel', ktrain, ktest};
	y=check_accuracy(accuracy, data);
