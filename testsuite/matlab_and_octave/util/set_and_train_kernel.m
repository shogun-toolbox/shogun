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
	if findstr('SIGMOID', kname)
		global kernel_arg1_gamma_;
		global kernel_arg2_coef0;
		sg('set_kernel', kname, ftype, size_cache, kernel_arg1_gamma_, kernel_arg2_coef0);
	elseif findstr('CHI2', kname)
		global kernel_arg0_width;
		sg('set_kernel', kname, ftype, size_cache, kernel_arg0_width);
	elseif findstr('CONST', kname)
		global kernel_arg0_c;
		sg('set_kernel', kname, ftype, size_cache, kernel_arg0_c);
	elseif findstr('DIAG', kname)
		global kernel_arg0_diag;
		sg('set_kernel', kname, ftype, size_cache, kernel_arg0_diag);
	elseif findstr('GAUSSIANSHIFT', kname)
		global kernel_arg0_width;
		global kernel_arg1_max_shift;
		global kernel_arg2_shift_step;
		sg('set_kernel', kname, ftype, size_cache, kernel_arg0_width, kernel_arg1_max_shift, kernel_arg2_shift_step);
	elseif findstr('GAUSSIAN', kname)
		global kernel_arg0_width;
		sg('set_kernel', kname, ftype, size_cache, kernel_arg0_width);
	elseif findstr('LINEAR', kname)
		global kernel_arg0_scale;
		sg('set_kernel', kname, ftype, size_cache, kernel_arg0_scale);
	elseif findstr('POLY', kname)
		global kernel_arg0_degree;
		global kernel_arg1_inhomogene;
		global kernel_arg2_use_normalization;

		sg('set_kernel', kname, ftype, size_cache, kernel_arg0_degree,
			eval(tolower(kernel_arg1_inhomogene)),
			eval(tolower(kernel_arg2_use_normalization)));
	elseif findstr('COMMSTRING', kname)
		global kernel_arg1_normalization;
		norm=fix_normalization_inconsistency(kernel_arg1_normalization);
		sg('set_kernel', kname, ftype, size_cache, kernel_arg1_normalization);
	elseif findstr('DISTANCE', kname)
		global kernel_arg1_distance;
		global kernel_arg0_width;
		dname=fix_distance_name_inconsistency(kernel_arg1_distance);
		% FIXME: REAL is cheating and will break in the future
		sg('set_distance', dname, 'REAL');
		sg('set_kernel', kname, size_cache, kernel_arg0_width);
	else
		printf('Unknown kernel %s', kname);
		y=1;
	end

	sg('init_kernel', 'TRAIN');
	y=0;
