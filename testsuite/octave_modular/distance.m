function y = distance(filename)
	init_shogun;
	y=true;

	addpath('util');
	addpath('../data/distance');

	eval('globals'); % ugly hack to have vars from filename as globals
	eval(filename);

	if ~set_features('distance_')
		return;
	end
	if ~set_distance()
		return;
	end

	dm_train=max(max(abs(distance_matrix_train-distance.get_distance_matrix())));
	distance.init(feats_train, feats_test);
	dm_test=max(max(abs(distance_matrix_test-distance.get_distance_matrix())));

	data={'distance', dm_train, dm_test};
	y=check_accuracy(distance_accuracy, data);
