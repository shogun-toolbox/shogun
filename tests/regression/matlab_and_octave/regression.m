function y = regression(filename)
	addpath('util');
	addpath('../data/regression');
	y=true;

	eval('globals'); % ugly hack to have vars from filename as globals
	eval(filename);

	if ~set_features('kernel_')
		return;
	end

	if ~set_kernel()
		return;
	end

	sg('threads', regression_num_threads);
	sg('set_labels', 'TRAIN', regression_labels);

	rname=fix_regression_name_inconsistency(regression_name);
	try
		sg('new_regression', rname);
	catch
		fprintf('Cannot set regression %s!\n', rname);
		return;
	end

	if strcmp(regression_type, 'svm')==1
		sg('c', regression_C);
		sg('svm_epsilon', regression_epsilon);
		sg('svr_tube_epsilon', regression_tube_epsilon);
	elseif strcmp(regression_type, 'kernelmachine')==1
		sg('krr_tau', regression_tau);
	else
		error('Incomplete regression data!\n');
	end

	sg('train_regression');

	alphas=0;
	bias=0;
	sv=0;
	if ~isempty(regression_bias)
		[bias, weights]=sg('get_svm');
		bias=abs(bias-regression_bias);
		weights=weights';
		for i = 1:length(weights(1:1, :))
			alphas=alphas+weights(1:1, i:i);
		end
		alphas=abs(alphas-regression_alpha_sum);
		for i = 1:length(weights(2:2, :))
			sv=sv+weights(2:2, i:i);
		end
		sv=abs(sv-regression_sv_sum);
	end

	classified=max(abs(sg('classify')-regression_classified));

	data={'classifier', alphas, bias, sv, classified};
	y=check_accuracy(regression_accuracy, data);
