function y = set_kernel()
	global kernel_name;
	global kernel_feature_type;
	global kernel_arg0_size;
	global kernel_arg1_size;
	global kernel_normalizer;
	y=false;

	if findstr('Sparse', kernel_name)
		is_sparse=true;
	else
		is_sparse=false;
	end
	kname=fix_kernel_name_inconsistency(kernel_name);

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


	if ~isempty(kernel_feature_type)
		ftype=upper(kernel_feature_type);
	else
		ftype='UNKNOWN';
	end

	% this endless list sux, but dunno how to do it differently here :(
	if strcmp(kname, 'SIGMOID')==1
		global kernel_arg1_gamma;
		global kernel_arg2_coef0;
		sg('set_kernel', kname, ftype, size_cache, kernel_arg1_gamma, kernel_arg2_coef0);

	elseif strcmp(kname, 'CHI2')==1
		global kernel_arg1_width;
		sg('set_kernel', kname, ftype, size_cache, kernel_arg1_width);

	elseif strcmp(kname, 'CONST')==1
		global kernel_arg0_c;
		sg('set_kernel', kname, ftype, size_cache, kernel_arg0_c);

	elseif strcmp(kname, 'DIAG')==1
		global kernel_arg1_diag;
		sg('set_kernel', kname, ftype, size_cache, kernel_arg1_diag);

	elseif strcmp(kname, 'GAUSSIANSHIFT')==1
		global kernel_arg1_width;
		global kernel_arg2_max_shift;
		global kernel_arg3_shift_step;
		sg('set_kernel', kname, ftype, size_cache, kernel_arg1_width, kernel_arg2_max_shift, kernel_arg3_shift_step);

	elseif strcmp(kname, 'GAUSSIAN')==1
		global kernel_arg0_width;
		global kernel_arg1_width;
		if ~isempty(kernel_arg0_width)
			width=kernel_arg0_width;
		else
			width=kernel_arg1_width;
		end
		if is_sparse
			ftype=strcat('SPARSE', ftype);
		end
		sg('set_kernel', kname, ftype, size_cache, width);

	elseif strcmp(kname, 'LINEAR')==1
		global kernel_arg0_scale;
		if isempty(kernel_arg0_scale)
			kernel_arg0_scale=-1;
		end
		if is_sparse
			ftype=strcat('SPARSE', ftype);
		end
		sg('set_kernel', kname, ftype, size_cache, kernel_arg0_scale);

	elseif strcmp(kname, 'POLYMATCH')==1 || strcmp(kname, 'POLYMATCHWORD')==1
		global kernel_arg1_degree;
		global kernel_arg2_inhomogene;
		sg('set_kernel', kname, ftype, size_cache, ...
			kernel_arg1_degree, tobool(kernel_arg2_inhomogene));

	elseif strcmp(kname, 'POLY')==1
		if is_sparse
			global kernel_arg1_degree;
			global kernel_arg2_inhomogene;
			degree=kernel_arg1_degree;
			inhomogene=kernel_arg2_inhomogene;
			ftype=strcat('SPARSE', ftype);
			sg('set_kernel', kname, ftype, size_cache, ...
				degree, tobool(inhomogene));
		else
			global kernel_arg1_degree;
			global kernel_arg2_inhomogene;
			degree=kernel_arg1_degree;
			inhomogene=tobool(kernel_arg2_inhomogene);
			sg('set_kernel', kname, ftype, size_cache, degree, inhomogene);
		end

	elseif strcmp(kname, 'MATCH')==1
		global kernel_arg1_degree;
		sg('set_kernel', kname, ftype, size_cache, kernel_arg1_degree);

	elseif findstr(kname, 'COMMSTRING') % normal + WEIGHTED
		global kernel_arg1_use_sign;
		if ~isempty(kernel_arg1_use_sign)
			sg('set_kernel', kname, ftype, size_cache, ...
				tobool(kernel_arg1_use_sign));
		else
			sg('set_kernel', kname, ftype, size_cache);
		end

	elseif findstr(kname, 'DEGREE') % FIXED + WEIGHTED
		global kernel_arg0_degree;
		global kernel_arg1_degree;
		if ~isempty(kernel_arg0_degree)
			degree=kernel_arg0_degree;
		else
			degree=kernel_arg1_degree;
		end
		sg('set_kernel', kname, ftype, size_cache, degree);

	elseif strcmp(kname, 'HISTOGRAM')==1 || strcmp(kname, 'SALZBERG')==1
		global classifier_labels;
		pseudo_pos=1e-10;
		pseudo_neg=1e-10;
		sg('new_plugin_estimator', pseudo_pos, pseudo_neg);
		sg('set_labels', 'TRAIN', double(classifier_labels));
		sg('train_estimator');

		sg('set_kernel', kname, ftype, size_cache);

	elseif strcmp(kname, 'DISTANCE')==1
		global kernel_arg2_distance;
		global kernel_arg1_width;
		dname=fix_distance_name_inconsistency(kernel_arg2_distance);
		% FIXME: REAL is cheating and will break in the future
		sg('set_distance', dname, 'REAL');
		sg('set_kernel', kname, size_cache, kernel_arg1_width);

	elseif strcmp(kname, 'LOCALALIGNMENT')==1
		sg('set_kernel', kname, ftype, size_cache);

	elseif findstr(kname, 'LIK')
		global kernel_arg1_length;
		global kernel_arg2_inner_degree;
		global kernel_arg3_outer_degree;
		sg('set_kernel', kname, ftype, size_cache, ...
			kernel_arg1_length, ...
			kernel_arg2_inner_degree, ...
			kernel_arg3_outer_degree);

	elseif strcmp(kname, 'OLIGO')==1
		global kernel_arg1_k;
		global kernel_arg2_width;
		sg('set_kernel', kname, ftype, size_cache, ...
			kernel_arg1_k, kernel_arg2_width);

	elseif strcmp(kname, 'COMBINED')
		% this will break when test file is changed!
		global kernel_subkernel0_name;
		global kernel_subkernel0_feature_type;
		global kernel_subkernel0_arg0_size;
		global kernel_subkernel0_arg1_degree;
		global kernel_subkernel1_name;
		global kernel_subkernel1_feature_type;
		global kernel_subkernel1_arg0_size;
		global kernel_subkernel1_arg1_degree;
		global kernel_subkernel1_arg2_inhomogene;
		global kernel_subkernel2_name;
		global kernel_subkernel2_feature_type;
		global kernel_subkernel2_arg0_size;

		sg('set_kernel', 'COMBINED', size_cache);

		subkernel_name=fix_kernel_name_inconsistency(kernel_subkernel0_name);
		sg('add_kernel', 1., subkernel_name, ...
			upper(kernel_subkernel0_feature_type), ...
			kernel_subkernel0_arg0_size, ...
			kernel_subkernel0_arg1_degree);

		subkernel_name=fix_kernel_name_inconsistency(kernel_subkernel1_name);
		sg('add_kernel', 1., subkernel_name, ...
			upper(kernel_subkernel1_feature_type), ...
			kernel_subkernel1_arg0_size, ...
			kernel_subkernel1_arg1_degree, ...
			tobool(kernel_subkernel1_arg2_inhomogene));

		subkernel_name=fix_kernel_name_inconsistency(kernel_subkernel2_name);
		sg('add_kernel', 1., subkernel_name, ...
			upper(kernel_subkernel2_feature_type), ...
			kernel_subkernel2_arg0_size);

	else
		error('Unknown kernel %s!\n', kname);
	end

	if ~isempty(kernel_normalizer)
		nname=fix_normalizer_name_inconsistency(kernel_normalizer);
		sg('set_kernel_normalization', nname);
	end
	y=true;
