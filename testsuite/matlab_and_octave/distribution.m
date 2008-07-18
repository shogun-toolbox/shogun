function y = distribution(filename)
	addpath('util');
	addpath('../data/distribution');

	eval('globals'); % ugly hack to have vars from filename as globals
	eval(filename);

	if !set_features()
		y=false;
		return;
	end

	if strcmp(name, 'HMM')==1
		sg('new_hmm', distribution_N, distribution_M);
		sg('bw');
	else
		printf("Can\'t yet train other distributions than HMM in static interface.\n");
		y=true;
		return;
	end

	if !set_and_train_distance()
		y=false;
		return;
	end

	likelihood=abs(sg('hmm_likelihood')-distribution_likelihood);
	y=check_accuracy_distribution(distribution_accuracy, likelihood);
