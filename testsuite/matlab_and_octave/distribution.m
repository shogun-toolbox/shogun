function y = distribution(filename)
	addpath('util');
	addpath('../data/distribution');
	y=true;

	eval('globals'); % ugly hack to have vars from filename as globals
	eval(filename);

	sg('init_random', init_random);
	rand('state', init_random);

	if ~set_features('distribution_')
		return;
	end


	if strcmp(distribution_name, 'HMM')==1
		sg('new_hmm', distribution_N, distribution_M);
		sg('bw');
	else
		fprintf('Cannot yet train other distributions than HMM!\n');
		return;
	end

	likelihood=abs(sg('hmm_likelihood')-distribution_likelihood);

	data={'distribution', likelihood, 0};
	y=check_accuracy(distribution_accuracy, data);
