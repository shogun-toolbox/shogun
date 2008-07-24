function y = test_distance(filename)
	init_shogun;
	y=true;

	addpath('util');
	addpath('../data/distance');

	eval('globals'); % ugly hack to have vars from filename as globals
	eval(filename);

	if !set_features()
		return;
	end
	if !set_distance()
		return;
	end

	dtrain=max(max(abs(dm_train-dist.get_distance_matrix())));
	dist.init(feats_train, feats_test);
	dtest=max(max(abs(dm_test-dist.get_distance_matrix())));

	y=check_accuracy(accuracy, dtrain, dtest);
