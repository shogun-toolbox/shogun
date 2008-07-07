function y = set_and_train_kernel()
	global kernel_name;
	global name_kernel;
	global name;
	global feature_type;
	global kernel_arg0_size;

	if !isempty(kernel_name)
		kname=fix_kernel_name_inconsistency(kernel_name);
	elseif !isempty(name_kernel)
		kname=fix_kernel_name_inconsistency(name_kernel);
	elseif !isempty(name)
		kname=fix_kernel_name_inconsistency(name);
	else
		disp('Something is wrong with the input data!')
		y=1;
		return
	end

	if !isempty(kernel_arg0_size)
		size_cache=kernel_arg0_size;
	else
		size_cache=10;
	end


	% this sux awfully, but dunno how to do it differently here :(
	if findstr('SIGMOID', kname)
		global kernel_arg1_gamma_;
		global kernel_arg2_coef0;
		sg('set_kernel', kname, toupper(feature_type), size_cache, kernel_arg1_gamma_, kernel_arg2_coef0);
	elseif findstr('COMMSTRING', kname)
		global kernel_arg1_normalization;
		norm=fix_normalization_inconsistency(kernel_arg1_normalization);
		sg('set_kernel', kname, toupper(feature_type), size_cache, kernel_arg1_normalization);
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
