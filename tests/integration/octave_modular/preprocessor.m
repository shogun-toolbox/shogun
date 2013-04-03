function y = preprocessor(filename)
	init_shogun;
	y=true;

	addpath('util');
	addpath('../data/preprocessor');

	eval('globals'); % ugly hack to have vars from filename as globals
	eval(filename);

	if strcmp(preprocessor_name, 'LogPlusOne')==1
		preproc=LogPlusOne();
	elseif strcmp(preprocessor_name, 'NormOne')==1
		preproc=NormOne();
	elseif strcmp(preprocessor_name, 'PruneVarSubMean')==1
		preproc=PruneVarSubMean(tobool(preprocessor_arg0_divide));
	elseif strcmp(preprocessor_name, 'SortUlongString')==1
		preproc=SortUlongString();
	elseif strcmp(preprocessor_name, 'SortWordString')==1
		preproc=SortWordString();
	else
		error('Unsupported preproc %s', preprocessor_name);
	end

	if ~set_features('kernel_')
		return;
	end

	preproc.init(feats_train);
	feats_train.add_preprocessor(preproc);
	feats_train.apply_preprocessor();
	feats_test.add_preprocessor(preproc);
	feats_test.apply_preprocessor();

	if ~set_kernel()
		return;
	end

	km_train=max(max(abs(kernel_matrix_train-kernel.get_kernel_matrix())));
	kernel.init(feats_train, feats_test);
	km_test=max(max(abs(kernel_matrix_test-kernel.get_kernel_matrix())));

	data={'kernel', km_train, km_test};
	y=check_accuracy(kernel_accuracy, data);
