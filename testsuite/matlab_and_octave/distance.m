function y = distance(filename)
	addpath('util');
	addpath('../data/distance');
	y=true;

	eval('globals'); % ugly hack to have vars from filename as globals
	eval(filename);

	if ~set_features()
		return;
	end

	if ~set_distance()
		return;
	end

	dmatrix=sg('get_distance_matrix');
	dtrain=max(max(abs(dm_train-dmatrix)));

	sg('init_distance', 'TEST');
	dmatrix=sg('get_distance_matrix');
	dtest=max(max(abs(dm_test-dmatrix)));

	data={'distance', dtrain, dtest};
	y=check_accuracy(accuracy, data);
