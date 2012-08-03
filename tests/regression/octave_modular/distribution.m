function y = distribution(filename)
	init_shogun;
	y=true;
	addpath('util');
	addpath('../data/distribution');

	eval('globals'); % ugly hack to have vars from filename as globals
	eval(filename);

	Math_init_random(init_random);

	if ~set_features('distribution_')
		return;
	end

	if strcmp(distribution_name, 'Histogram')==1
		distribution=Histogram(feats_train);
		distribution.train();

	elseif strcmp(distribution_name, 'HMM')==1
		global BW_NORMAL;
		distribution=HMM(feats_train,
			distribution_N, distribution_M, distribution_pseudo);
		distribution.train();
		distribution.baum_welch_viterbi_train(BW_NORMAL);

	elseif strcmp(distribution_name, 'LinearHMM')==1
		distribution=LinearHMM(feats_train);
		distribution.train();

	else
		error('Unsupported distribution %s!', distribution_name);
	end


	likelihood=max(abs(distribution.get_log_likelihood_sample()-distribution_likelihood));

	num_examples=feats_train.get_num_vectors();
	num_param=distribution.get_num_model_parameters();
	derivatives=0;
	for i = 0:num_param-1
		for j = 0:num_examples-1
			val=distribution.get_log_derivative(i, j);
			if val!=-Inf && val!=NaN % only consider sparse matrix!
				derivatives+=val;
			end
		end
	end
	derivatives=max(abs(derivatives-distribution_derivatives));

	data={'distribution', likelihood, derivatives};
	y=check_accuracy(distribution_accuracy, data);
