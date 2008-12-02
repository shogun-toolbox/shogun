function y = kernel(filename)
	init_shogun;
	y=true;
	addpath('util');
	addpath('../data/kernel');

	eval('globals'); % ugly hack to have vars from filename as globals
	eval(filename);

	if ~set_features('kernel_')
		return;
	end
	if ~set_kernel()
		return;
	end

	if strcmp(kernel_name, 'Custom')==1
		kmt=[];
		sz=size(kernel_matrix_triangletriangle, 2);
		for i=1:size(kernel_matrix_triangletriangle, 2)
			kmt=[kmt kernel_matrix_triangletriangle(i, i:sz)];
		end

		kernel.set_triangle_kernel_matrix_from_triangle(kmt);
		triangletriangle=max(max(abs(kernel_matrix_triangletriangle-kernel.get_kernel_matrix())));

		kernel.set_triangle_kernel_matrix_from_full(kernel_matrix_fulltriangle);
		fulltriangle=max(max(abs(kernel_matrix_fulltriangle-kernel.get_kernel_matrix())));

		kernel.set_full_kernel_matrix_from_full(kernel_matrix_fullfull);
		fullfull=max(max(abs(kernel_matrix_fullfull-kernel.get_kernel_matrix())));

		data={'custom', triangletriangle, fulltriangle, fullfull};
		y=check_accuracy(kernel_accuracy, data);
	else
		km_train=max(max(abs(kernel_matrix_train-kernel.get_kernel_matrix())));
		kernel.init(feats_train, feats_test);
		km_test=max(max(abs(kernel_matrix_test-kernel.get_kernel_matrix())));

		data={'kernel', km_train, km_test};
		y=check_accuracy(kernel_accuracy, data);
	end
