function y = distance(filename)
	addpath('util');
	addpath('../data/distance');
	y=true;

	eval('globals'); % ugly hack to have vars from filename as globals
	eval(filename);

	if ~set_features('distance_')
		return;
	end

	if ~set_distance()
		return;
	end

	dmatrix=sg('get_distance_matrix', 'TRAIN');
	dm_train=max(max(abs(distance_matrix_train-dmatrix)));

	dmatrix=sg('get_distance_matrix', 'TEST');
	dm_test=max(max(abs(distance_matrix_test-dmatrix)));

	data={'distance', dm_train, dm_test};
	y=check_accuracy(distance_accuracy, data);
