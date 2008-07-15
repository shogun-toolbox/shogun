function y = set_and_train_kernel()
	global kernel_name;
	global name;
	global feature_type;
	global kernel_arg0_size;
	global kernel_arg1_size;

	if !isempty(kernel_name)
		kname=fix_kernel_name_inconsistency(kernel_name);
	elseif !isempty(name)
		kname=fix_kernel_name_inconsistency(name);
	else
		disp('Something is wrong with the input data!')
		y=1;
		return
	end

	if !isempty(kernel_arg0_size)
		size_cache=kernel_arg0_size;
	elseif !isempty(kernel_arg1_size)
		size_cache=kernel_arg1_size;
	else
		size_cache=10;
	end


	% this sux awfully, but dunno how to do it differently here :(
	ftype=toupper(feature_type);
	if strcmp(kname, 'SIGMOID')==1
		global kernel_arg1_gamma_;
		global kernel_arg2_coef0;
		sg('set_kernel', kname, ftype, size_cache, kernel_arg1_gamma_, kernel_arg2_coef0);

	elseif strcmp(kname, 'CHI2')==1
		global kernel_arg0_width;
		sg('set_kernel', kname, ftype, size_cache, kernel_arg0_width);

	elseif strcmp(kname, 'CONST')==1
		global kernel_arg0_c;
		sg('set_kernel', kname, ftype, size_cache, kernel_arg0_c);

	elseif strcmp(kname, 'DIAG')==1
		global kernel_arg0_diag;
		sg('set_kernel', kname, ftype, size_cache, kernel_arg0_diag);

	elseif strcmp(kname, 'GAUSSIANSHIFT')==1
		global kernel_arg0_width;
		global kernel_arg1_max_shift;
		global kernel_arg2_shift_step;
		sg('set_kernel', kname, ftype, size_cache, kernel_arg0_width, kernel_arg1_max_shift, kernel_arg2_shift_step);

	elseif strcmp(kname, 'GAUSSIAN')==1
		global kernel_arg0_width;
		sg('set_kernel', kname, ftype, size_cache, kernel_arg0_width);

	elseif strcmp(kname, 'LINEAR')==1
		global kernel_arg0_scale;
		if isempty(kernel_arg0_scale)
			kernel_arg0_scale=-1;
		end
		sg('set_kernel', kname, ftype, size_cache, kernel_arg0_scale);

	elseif strcmp(kname, 'POLYMATCH')==1
		global kernel_arg0_degree;
		global kernel_arg1_inhomogene;
		sg('set_kernel', kname, ftype, size_cache, kernel_arg0_degree,
			eval(tolower(kernel_arg1_inhomogene)));

	elseif strcmp(kname, 'POLY')==1
		global kernel_arg0_degree;
		global kernel_arg1_inhomogene;
		global kernel_arg2_use_normalization;
		sg('set_kernel', kname, ftype, size_cache, kernel_arg0_degree,
			eval(tolower(kernel_arg1_inhomogene)),
			eval(tolower(kernel_arg2_use_normalization)));

	elseif findstr(kname, 'COMMSTRING') % normal + WEIGHTED
		global kernel_arg0_use_sign;
		global kernel_arg1_normalization;
		norm=fix_normalization_inconsistency(kernel_arg1_normalization);
		sg('set_kernel', kname, ftype, size_cache,
				eval(tolower(kernel_arg0_use_sign)), norm);

	elseif findstr(kname, 'DEGREE') % FIXED + WEIGHTED
		global kernel_arg0_degree;
		sg('set_kernel', kname, ftype, size_cache, kernel_arg0_degree);

	elseif strcmp(kname, 'HISTOGRAM')==1 || strcmp(kname, 'SALZBERG')==1
		global classifier_labels;
		pseudo_pos=1e-10;
		pseudo_neg=1e-10;
		sg('new_plugin_estimator', pseudo_pos, pseudo_neg);
		sg('set_labels', 'TRAIN', double(classifier_labels));
		sg('train_estimator');

		sg('set_kernel', kname, ftype, size_cache);

	elseif strcmp(kname, 'DISTANCE')==1
		global kernel_arg1_distance;
		global kernel_arg0_width;
		dname=fix_distance_name_inconsistency(kernel_arg1_distance);
		% FIXME: REAL is cheating and will break in the future
		sg('set_distance', dname, 'REAL');
		sg('set_kernel', kname, size_cache, kernel_arg0_width);

	elseif strcmp(kname, 'LOCALALIGNMENT')==1
		sg('set_kernel', kname, ftype, size_cache);

	elseif findstr(kname, 'LIK')
		global kernel_arg0_length;
		global kernel_arg1_inner_degree;
		global kernel_arg2_outer_degree;
		sg('set_kernel', kname, ftype, size_cache, kernel_arg0_length,
			kernel_arg1_inner_degree, kernel_arg2_outer_degree);

	elseif strcmp(kname, 'AUC')==1 || strcmp(kname, 'CUSTOM')==1
		printf("Kernel %s yet unsupported in static interfaces.\n", kname);
		y=1;
		return

	else
		printf("Unknown kernel %s.\n", kname);
		y=1;
		return
	end

	sg('init_kernel', 'TRAIN');
	y=0;
