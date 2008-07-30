function y = distance(filename)
	init_shogun;
	y=true;

	addpath('util');
	addpath('../data/distance');

	eval('globals'); % ugly hack to have vars from filename as globals
	eval(filename);

	if ~set_features()
		return;
	end
	if ~set_distance()
		return;
	end

	dtrain=max(max(abs(dm_train-distance.get_distance_matrix())));
	distance.init(feats_train, feats_test);
	dtest=max(max(abs(dm_test-distance.get_distance_matrix())));

	data={'distance', dtrain, dtest};
	y=check_accuracy(accuracy, data);
