function y = test_regression(filename)
	init_shogun;
	y=true;
	addpath('util');
	addpath('../data/regression');

	eval('globals'); % ugly hack to have vars from filename as globals
	eval(filename);

	if !set_features()
		return;
	end
	if !set_kernel()
		return;
	end
	kern.parallel.set_num_threads(regression_num_threads);

	lab=Labels(regression_labels);

	if strcmp(name, 'KRR')==1
		regression=KRR(regression_tau, kern, lab);
	elseif strcmp(name, 'LibSVR')==1
		regression=LibSVR(regression_C, regression_epsilon, kern, lab);
		regression.set_tube_epsilon(regression_tube_epsilon);
	elseif strcmp(name, 'SVRLight')==1
		regression=SVRLight(regression_C, regression_epsilon, kern, lab);
		regression.set_tube_epsilon(regression_tube_epsilon);
	else
		error('Unsupported regression %s!', name);
	end

	regression.parallel.set_num_threads(regression_num_threads);
	regression.train();

	alphas=0;
	bias=0;
	sv=0;
	if !isempty(regression_bias)
		bias=regression.get_bias();
		bias=abs(bias-regression_bias);
		alphas=regression.get_alphas();
		alphas=max(abs(alphas-regression_alphas));
		sv=regression.get_support_vectors();
		sv=max(abs(sv-regression_support_vectors));
	end

	kern.init(feats_train, feats_test);
	classified=max(abs(
		regression.classify().get_labels()-regression_classified));

	y=check_accuracy_classifier(regression_accuracy, alphas, bias, sv, classified);
