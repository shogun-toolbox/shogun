function y = regression(filename)
	addpath('util');
	addpath('../data/regression');

	eval('globals'); % ugly hack to have vars from filename as globals
	eval(filename);

	if !set_features()
		y=false;
		return;
	end

	if !set_and_train_kernel()
		y=false;
		return;
	end

	sg('threads', regression_num_threads);
	sg('set_labels', 'TRAIN', regression_labels);

	rname=fix_regression_name_inconsistency(name);
	sg('new_regression', rname);

	if strcmp(regression_type, 'svm')==1
		sg('c', regression_C);
		sg('svm_epsilon', regression_epsilon);
		sg('svr_tube_epsilon', regression_tube_epsilon);
	elseif strcmp(regression_type, 'kernelmachine')==1
		sg('krr_tau', regression_tau);
	else
		printf("Incomplete regression data!\n");
	end

	sg('train_regression');

	alphas=0;
	bias=0;
	sv=0;
	if !isempty(regression_bias)
		[bias, weights]=sg('get_svm');
		bias=abs(bias-regression_bias);
		weights=weights';
		alphas=max(abs(weights(1:1,:)-regression_alphas));
		sv=max(abs(weights(2:2,:)-regression_support_vectors));
	end

	sg('init_kernel', 'TEST');
	classified=max(abs(sg('classify')-regression_classified));

	y=check_accuracy_classifier(regression_accuracy, alphas, bias, sv, classified);
