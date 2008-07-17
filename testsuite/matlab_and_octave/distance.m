function y = distance(filename)
	addpath('util');
	addpath('../data/distance');

	eval('globals'); % ugly hack to have vars from filename as globals
	eval(filename);

	if !set_features()
		y=false;
		return;
	end

	if !set_and_train_distance()
		y=false;
		return;
	end

	dmatrix=sg('get_distance_matrix');
	dtrain=max(abs(dm_train-dmatrix))(1:1);

	sg('init_distance', 'TEST');
	dmatrix=sg('get_distance_matrix');
	dtest=max(abs(dm_test-dmatrix))(1:1);

	y=check_accuracy(accuracy, dtrain, dtest);
