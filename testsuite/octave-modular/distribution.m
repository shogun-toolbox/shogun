function y = distribution(filename)
	y=true;
	addpath('util');
	addpath('../data/distribution');

	eval('globals'); % ugly hack to have vars from filename as globals
	eval(filename);

	if !set_features()
		return;
	end

	if strcmp(name, 'HMM')==1
		sg('new_hmm', distribution_N, distribution_M);
		sg('bw');
	elseif strcmp(name, 'Histogram')==1 || strcmp(name, 'LinearHMM')==1
		disp('Cannot yet train other distributions than HMM in static interface.');
		return;
	else
		error('Unsupported distribution %s', name);
	end

	likelihood=abs(sg('hmm_likelihood')-distribution_likelihood);
	y=check_accuracy_distribution(distribution_accuracy, likelihood);
