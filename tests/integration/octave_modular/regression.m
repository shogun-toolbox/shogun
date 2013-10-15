function y = regression(filename)
	init_shogun;
	y=true;
	addpath('util');
	addpath('../data/regression');

	eval('globals'); % ugly hack to have vars from filename as globals
	eval(filename);
	prefix='regression_';

	if ~set_features('kernel_')
		return;
	end
	if ~set_kernel()
		return;
	end
	kernel.parallel.set_num_threads(regression_num_threads);

	lab=RegressionLabels(regression_labels);

	if strcmp(regression_name, 'KERNELRIDGEREGRESSION')==1
		regression=KernelRidgeRegression(regression_tau, kernel, lab);

	elseif strcmp(regression_name, 'LibSVR')==1
		regression=LibSVR(regression_C, regression_epsilon, kernel, lab);
		regression.set_tube_epsilon(regression_tube_epsilon);

	elseif strcmp(regression_name, 'SVRLight')==1
		try
			regression=SVRLight(regression_C, regression_epsilon, kernel, lab);
			regression.set_tube_epsilon(regression_tube_epsilon);
		catch
			disp('No support for SVRLight available.');
			return;
		end

	else
		error('Unsupported regression %s!', regression_name);
	end

	regression.parallel.set_num_threads(regression_num_threads);
	regression.train();

	bias=0;
	if ~isempty(regression_bias)
		bias=regression.get_bias();
		bias=abs(bias-regression_bias);
	end

	alphas=0;
	sv=0;
	if ~isempty(regression_alpha_sum) && ~isempty(regression_sv_sum)
			tmp=regression.get_alphas();
			for i = 1:length(tmp)
				alphas=alphas+tmp(i:i);
			end
			alphas=abs(alphas-regression_alpha_sum);
			tmp=regression.get_support_vectors();
			for i = 1:length(tmp)
				sv=sv+tmp(i:i);
			end
			sv=abs(sv-regression_sv_sum);
	end

	kernel.init(feats_train, feats_test);
	classified=max(abs(
		regression.apply().get_labels()-regression_classified));

	data={'classifier', alphas, bias, sv, classified};
	y=check_accuracy(regression_accuracy, data);
