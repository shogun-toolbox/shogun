function y = test_kernel(filename)
	init_shogun;
	y=true;
	addpath('util');
	addpath('../data/kernel');

	eval('globals'); % ugly hack to have vars from filename as globals
	eval(filename);

	if !set_features()
		return;
	end
	if !set_kernel()
		return;
	end

	if strcmp(name, 'Custom')==1
		kern.io.set_loglevel(M_DEBUG)
		kern.set_triangle_kernel_matrix_from_triangle(tril(symdata));
		triangletriangle=max(abs(km_triangletriangle-kern.get_kernel_matrix()));

		%kern.set_triangle_kernel_matrix_from_full(symdata);
		%fulltriangle=max(abs(km_fulltriangle-kern.get_kernel_matrix()));
		%kern.set_full_kernel_matrix_from_full(data)
		%fullfull=max(abs(km_fullfull-kern.get_kernel_matrix()))

		%y=check_accuracy_custom(accuracy,
		%	triangletriangle, fulltriangle, fullfull);
	else
		ktrain=max(max(abs(km_train-kern.get_kernel_matrix())));
		kern.init(feats_train, feats_test);
		ktest=max(max(abs(km_test-kern.get_kernel_matrix())));

		y=check_accuracy(accuracy, ktrain, ktest);
	end
