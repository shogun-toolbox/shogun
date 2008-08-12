function y = kernel(filename)
	init_shogun;
	y=true;
	addpath('util');
	addpath('../data/kernel');

	eval('globals'); % ugly hack to have vars from filename as globals
	eval(filename);

	if ~set_features()
		return;
	end
	if ~set_kernel()
		return;
	end

	if strcmp(name, 'Custom')==1
		kmt=[];
		sz=size(km_triangletriangle, 2);
		for i=1:size(km_triangletriangle, 2)
			kmt=[kmt km_triangletriangle(i, i:sz)];
		end

		kernel.set_triangle_kernel_matrix_from_triangle(kmt);
		triangletriangle=max(max(abs(km_triangletriangle-kernel.get_kernel_matrix())));

		kernel.set_triangle_kernel_matrix_from_full(km_fulltriangle);
		fulltriangle=max(max(abs(km_fulltriangle-kernel.get_kernel_matrix())));

		kernel.set_full_kernel_matrix_from_full(km_fullfull);
		fullfull=max(max(abs(km_fullfull-kernel.get_kernel_matrix())));

		data={'custom', triangletriangle, fulltriangle, fullfull};
		y=check_accuracy(accuracy, data);
	else
		ktrain=max(max(abs(km_train-kernel.get_kernel_matrix())));
		kernel.init(feats_train, feats_test);
		ktest=max(max(abs(km_test-kernel.get_kernel_matrix())));

		data={'kernel', ktrain, ktest};
		y=check_accuracy(accuracy, data);
	end
