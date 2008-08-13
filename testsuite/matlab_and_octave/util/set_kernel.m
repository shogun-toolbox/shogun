function y = set_kernel()
	global kernel_name;
	global name;
	global feature_type;
	global kernel_arg0_size;
	global kernel_arg1_size;
	y=false;

	is_sparse=false;
	if ~isempty(kernel_name)
		kname=fix_kernel_name_inconsistency(kernel_name);
	else
		if findstr('Sparse', name)
			is_sparse=true;
		end
		kname=fix_kernel_name_inconsistency(name);
	end

	if strcmp(kname, 'AUC')==1 || strcmp(kname, 'CUSTOM')==1
		fprintf('Kernel %s not supported yet!\n', kname);
		return;
	end

	if ~isempty(kernel_arg0_size)
		size_cache=kernel_arg0_size;
	elseif ~isempty(kernel_arg1_size)
		size_cache=kernel_arg1_size;
	else
		size_cache=10;
	end


	% this sux awfully, but dunno how to do it differently here :(
	ftype=upper(feature_type);
	if strcmp(kname, 'SIGMOID')==1
		global kernel_arg1_gamma;
		global kernel_arg2_coef0;
		sg('set_kernel', kname, ftype, size_cache, kernel_arg1_gamma, kernel_arg2_coef0);

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
		if is_sparse
			ftype=strcat('SPARSE', ftype);
		end
		sg('set_kernel', kname, ftype, size_cache, kernel_arg0_width);

	elseif strcmp(kname, 'LINEAR')==1
		global kernel_arg0_scale;
		if isempty(kernel_arg0_scale)
			kernel_arg0_scale=-1;
		end
		if is_sparse
			ftype=strcat('SPARSE', ftype);
		end
		sg('set_kernel', kname, ftype, size_cache, kernel_arg0_scale);

	elseif strcmp(kname, 'POLYMATCH')==1
		global kernel_arg0_degree;
		global kernel_arg1_inhomogene;
		sg('set_kernel', kname, ftype, size_cache, ...
			kernel_arg0_degree, tobool(kernel_arg1_inhomogene));

	elseif strcmp(kname, 'POLY')==1
		if is_sparse
			global kernel_arg1_degree;
			global kernel_arg2_inhomogene;
			global kernel_arg3_use_normalization;
			degree=kernel_arg1_degree;
			inhomogene=kernel_arg2_inhomogene;
			use_normalization=kernel_arg3_use_normalization;
			ftype=strcat('SPARSE', ftype);
		else
			global kernel_arg0_degree;
			global kernel_arg1_inhomogene;
			global kernel_arg2_use_normalization;
			degree=kernel_arg0_degree;
			inhomogene=kernel_arg1_inhomogene;
			use_normalization=kernel_arg2_use_normalization;
		end

		sg('set_kernel', kname, ftype, size_cache, ...
			degree, tobool(inhomogene), tobool(use_normalization));

	elseif strcmp(kname, 'MATCH')==1
		global kernel_arg0_degree;
		sg('set_kernel', kname, ftype, size_cache, kernel_arg0_degree);

	elseif findstr(kname, 'COMMSTRING') % normal + WEIGHTED
		global kernel_arg0_use_sign;
		global kernel_arg1_normalization;
		norm=fix_normalization_inconsistency(kernel_arg1_normalization);
		sg('set_kernel', kname, ftype, size_cache, ...
				tobool(kernel_arg0_use_sign), norm);

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
		sg('set_kernel', kname, ftype, size_cache, ...
			kernel_arg0_length, kernel_arg1_inner_degree, ...
			kernel_arg2_outer_degree);

	elseif strcmp(kname, 'COMBINED')
		% this will break when test file is changed!
		global subkernel0_name;
		global subkernel0_feature_type;
		global subkernel0_kernel_arg0_size;
		global subkernel0_kernel_arg1_degree;
		global subkernel1_name;
		global subkernel1_feature_type;
		global subkernel1_kernel_arg0_size;
		global subkernel1_kernel_arg1_degree;
		global subkernel1_kernel_arg2_inhomogene;
		global subkernel2_name;
		global subkernel2_feature_type;
		global subkernel2_kernel_arg0_size;

		sg('set_kernel', 'COMBINED', size_cache);

		subkernel_name=fix_kernel_name_inconsistency(subkernel0_name);
		sg('add_kernel', 1., subkernel_name, ...
			upper(subkernel0_feature_type), ...
			subkernel0_kernel_arg0_size, ...
			subkernel0_kernel_arg1_degree);

		subkernel_name=fix_kernel_name_inconsistency(subkernel1_name);
		sg('add_kernel', 1., subkernel_name, ...
			upper(subkernel1_feature_type), ...
			subkernel1_kernel_arg0_size, ...
			subkernel1_kernel_arg1_degree, ...
			tobool(subkernel1_kernel_arg2_inhomogene));

		subkernel_name=fix_kernel_name_inconsistency(subkernel2_name);
		sg('add_kernel', 1., subkernel_name, ...
			upper(subkernel2_feature_type), ...
			subkernel2_kernel_arg0_size);

	else
		error('Unknown kernel %s!\n', kname);
	end

	sg('init_kernel', 'TRAIN');
	y=true;
