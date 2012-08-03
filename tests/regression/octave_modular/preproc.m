function y = preproc(filename)
	init_shogun;
	y=true;

	addpath('util');
	addpath('../data/preproc');

	eval('globals'); % ugly hack to have vars from filename as globals
	eval(filename);

	if strcmp(preproc_name, 'LogPlusOne')==1
		preproc=LogPlusOne();
	elseif strcmp(preproc_name, 'NormOne')==1
		preproc=NormOne();
	elseif strcmp(preproc_name, 'PruneVarSubMean')==1
		preproc=PruneVarSubMean(tobool(preproc_arg0_divide));
	elseif strcmp(preproc_name, 'SortUlongString')==1
		preproc=SortUlongString();
	elseif strcmp(preproc_name, 'SortWordString')==1
		preproc=SortWordString();
	else
		error('Unsupported preproc %s', preproc_name);
	end

	if ~set_features('kernel_')
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

	km_train=max(max(abs(kernel_matrix_train-kernel.get_kernel_matrix())));
	kernel.init(feats_train, feats_test);
	km_test=max(max(abs(kernel_matrix_test-kernel.get_kernel_matrix())));

	data={'kernel', km_train, km_test};
	y=check_accuracy(kernel_accuracy, data);
