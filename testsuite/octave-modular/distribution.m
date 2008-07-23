function y = distribution(filename)
	addpath('util');
	addpath('../data/distribution');

	eval('globals'); % ugly hack to have vars from filename as globals
	eval(filename);

	fset=set_features();
	if !fset
		y=false;
		return;
	elseif strcmp(fset, 'catchme')==1
		y=true;
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

	likelihood=abs(sg('hmm_likelihood')-distribution_likelihood);
	y=check_accuracy_distribution(distribution_accuracy, likelihood);
