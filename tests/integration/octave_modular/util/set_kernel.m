function y = set_kernel()
	global kernel_name;
	global feats_train;
	global feats_test;
	global kernel;
	global kernel_arg0_size;
	global kernel_arg1_size;
	global kernel_normalizer;
	global IdentityKernelNormalizer;
	global AvgDiagKernelNormalizer;
	global SqrtDiagKernelNormalizer;
	y=false;

	if !isempty(kernel_arg0_size)
		size_cache=kernel_arg0_size;
	elseif !isempty(kernel_arg1_size)
		size_cache=kernel_arg1_size;
	else
		size_cache=10;
	end

	if !isempty(kernel_name)
		kname=kernel_name;
	else
		kname=name;
	end

	% this sux awfully, but dunno how to do it differently here :(
	if strcmp(kname, 'AUC')==1
		global kernel_subkernel0_name;
		global kernel_subkernel0_data_train;
		global kernel_subkernel0_data_test;
		global kernel_subkernel0_arg1_width;
		global subkernel; % subkernel will be destroyed otherwise
		global RealFeatures;
		global GaussianKernel;
		global AUCKernel;

		% this will break when testcase is changed!
		if strcmp(kernel_subkernel0_name, 'Gaussian')!=1
			error('Dunno how to handle AUC subkernel %s', ...
				kernel_subkernel0_name);
		end

		subfeats_train=RealFeatures(kernel_subkernel0_data_train);
		subfeats_test=RealFeatures(kernel_subkernel0_data_test);
		subkernel=GaussianKernel(subfeats_train, subfeats_test, ...
			kernel_subkernel0_arg1_width);
		kernel=AUCKernel(0, subkernel);
		kernel.init(feats_train, feats_train);

		%feats_train=WordFeatures(kernel_subkernel0_data_train);
		%feats_test=WordFeatures(kernel_subkernel0_data_test);

	elseif strcmp(kname, 'Chi2')==1
		global Chi2Kernel;
		global kernel_arg1_width;
		kernel=Chi2Kernel(feats_train, feats_train, ...
			kernel_arg1_width, size_cache);

	elseif strcmp(kname, 'Combined')
		% this will break when test file is changed!
		global CombinedKernel;
		global FixedDegreeStringKernel;
		global PolyMatchStringKernel;
		global LocalAlignmentStringKernel;
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
		global StringCharFeatures;
		global DNA;

		kernel=CombinedKernel();
		subkernel=FixedDegreeStringKernel(...
			kernel_subkernel0_arg0_size, ...
			kernel_subkernel0_arg1_degree);
		kernel.append_kernel(subkernel);
		subkernel=PolyMatchStringKernel(kernel_subkernel1_arg0_size, ...
			kernel_subkernel1_arg1_degree, ...
			tobool(kernel_subkernel1_arg2_inhomogene));
		kernel.append_kernel(subkernel);
		subkernel=LocalAlignmentStringKernel(...
			kernel_subkernel2_arg0_size);
		kernel.append_kernel(subkernel);
		kernel.init(feats_train, feats_train);

	elseif strcmp(kname, 'CommUlongString')==1
		global CommUlongStringKernel;
		global kernel_arg1_use_sign;
		if ~isempty(kernel_arg1_use_sign)
			kernel=CommUlongStringKernel(feats_train, feats_train, ...
				tobool(kernel_arg1_use_sign));
		else
			kernel=CommUlongStringKernel(feats_train, feats_train);
		end

	elseif strcmp(kname, 'CommWordString')==1
		global CommWordStringKernel;
		global kernel_arg1_use_sign;
		if ~isempty(kernel_arg1_use_sign)
			kernel=CommWordStringKernel(feats_train, feats_train, ...
				tobool(kernel_arg1_use_sign));
		else
			kernel=CommWordStringKernel(feats_train, feats_train);
		end

	elseif strcmp(kname, 'Const')==1
		global ConstKernel;
		global kernel_arg0_c;
		kernel=ConstKernel(feats_train, feats_train, kernel_arg0_c);

	elseif strcmp(kname, 'Custom')==1
		global CustomKernel;
		kernel=CustomKernel();

	elseif strcmp(kname, 'Diag')==1
		global DiagKernel;
		global kernel_arg1_diag;
		kernel=DiagKernel(feats_train, feats_train, kernel_arg1_diag);

	elseif strcmp(kname, 'Distance')==1
		global DistanceKernel;
		global kernel_arg1_width;
		global distance;
		if !set_distance()
			y=false;
			return;
		end
		kernel=DistanceKernel(feats_train, feats_train, ...
			kernel_arg1_width, distance);

	elseif strcmp(kname, 'FixedDegreeString')==1
		global FixedDegreeStringKernel;
		global kernel_arg1_degree;
		kernel=FixedDegreeStringKernel(feats_train, feats_train, ...
			kernel_arg1_degree);

	elseif strcmp(kname, 'GaussianShift')==1
		global GaussianShiftKernel;
		global kernel_arg1_width;
		global kernel_arg2_max_shift;
		global kernel_arg3_shift_step;
		kernel=GaussianShiftKernel(feats_train, feats_train, ...
			kernel_arg1_width, kernel_arg2_max_shift, kernel_arg3_shift_step);

	elseif strcmp(kname, 'Gaussian')==1
		global GaussianKernel;
		global kernel_arg0_width;
		global kernel_arg1_width;
		if isempty(kernel_arg0_width)
			width=kernel_arg1_width;
		else
			width=kernel_arg0_width;
		end
		kernel=GaussianKernel(feats_train, feats_train, width);

	elseif strcmp(kname, 'HistogramWordString')==1
		global HistogramWordStringKernel;
		global pie;
		if !set_pie()
			return;
		end
		kernel=HistogramWordStringKernel(feats_train, feats_train, pie);

	elseif strcmp(kname, 'LinearByte')==1
		global LinearByteKernel;
		kernel=LinearByteKernel();
		kernel.set_normalizer(AvgDiagKernelNormalizer(-1));
		kernel.init(feats_train, feats_train);

	elseif strcmp(kname, 'LinearString')==1
		global LinearStringKernel;
		kernel=LinearStringKernel();
		kernel.set_normalizer(AvgDiagKernelNormalizer(-1));
		kernel.init(feats_train, feats_train);

	elseif strcmp(kname, 'LinearWord')==1
		global LinearWordKernel;
		kernel=LinearWordKernel();
		kernel.set_normalizer(AvgDiagKernelNormalizer(-1));
		kernel.init(feats_train, feats_train);

	elseif strcmp(kname, 'Linear')==1
		global LinearKernel;
		global kernel_arg0_scale;
		kernel=LinearKernel();
		if kernel_arg0_scale
			kernel.set_normalizer(AvgDiagKernelNormalizer(kernel_arg0_scale));
		else
			kernel.set_normalizer(AvgDiagKernelNormalizer(-1));
		end
		kernel.init(feats_train, feats_train);

	elseif strcmp(kname, 'LocalAlignmentString')==1
		global LocalAlignmentStringKernel;
		kernel=LocalAlignmentStringKernel(feats_train, feats_train);

	elseif strcmp(kname, 'OligoString')==1
		global OligoStringKernel;
		global kernel_arg1_k;
		global kernel_arg2_width;
		kernel=OligoStringKernel(size_cache, kernel_arg1_k, kernel_arg2_width);
		kernel.init(feats_train, feats_train);

	elseif strcmp(kname, 'PolyMatchString')==1
		global PolyMatchStringKernel;
		global kernel_arg1_degree;
		global kernel_arg2_inhomogene;
		kernel=PolyMatchStringKernel(feats_train, feats_train, ...
			kernel_arg1_degree, tobool(kernel_arg2_inhomogene));

	elseif strcmp(kname, 'PolyMatchWordString')==1
		global PolyMatchWordStringKernel;
		global kernel_arg1_degree;
		global kernel_arg2_inhomogene;
		kernel=PolyMatchWordStringKernel(feats_train, feats_train, ...
			kernel_arg1_degree, tobool(kernel_arg2_inhomogene));

	elseif strcmp(kname, 'Poly')==1
		global PolyKernel;
		global kernel_arg1_degree;
		global kernel_arg2_inhomogene;
		if ~isempty(kernel_normalizer)
			kernel=PolyKernel(size_cache, kernel_arg1_degree, ...
				tobool(kernel_arg2_inhomogene));
			kernel.set_normalizer(eval(sprintf([kernel_normalizer, '()'])));
			kernel.init(feats_train, feats_train);
		else
			kernel=PolyKernel(feats_train, feats_train, kernel_arg1_degree, ...
				tobool(kernel_arg2_inhomogene));
		end

	elseif strcmp(kname, 'SalzbergWordString')==1
		global SalzbergWordStringKernel;
		global pie;
		if !set_pie()
			return;
		end
		kernel=SalzbergWordStringKernel(feats_train, feats_train, pie);

	elseif strcmp(kname, 'Sigmoid')==1
		global SigmoidKernel;
		global kernel_arg1_gamma;
		global kernel_arg2_coef0;
		kernel=SigmoidKernel(feats_train, feats_train, size_cache, ...
			kernel_arg1_gamma, kernel_arg2_coef0);

	elseif strcmp(kname, 'SimpleLocalityImprovedString')==1
		global SimpleLocalityImprovedStringKernel;
		global kernel_arg1_length;
		global kernel_arg2_inner_degree;
		global kernel_arg3_outer_degree;
		kernel=SimpleLocalityImprovedStringKernel(feats_train, feats_train, ...
			kernel_arg1_length, kernel_arg2_inner_degree,
			kernel_arg3_outer_degree);

	elseif strcmp(kname, 'SparseGaussian')==1
		global SparseGaussianKernel;
		global kernel_arg1_width;
		kernel=SparseGaussianKernel(feats_train, feats_train, ...
			kernel_arg1_width);

	elseif strcmp(kname, 'SparseLinear')==1
		global SparseLinearKernel;
		global kernel_arg0_scale;
		kernel=SparseLinearKernel();
		if kernel_arg0_scale
			kernel.set_normalizer(AvgDiagKernelNormalizer(kernel_arg0_scale));
		else
			kernel.set_normalizer(AvgDiagKernelNormalizer(-1));
		end
		kernel.init(feats_train, feats_train);

	elseif strcmp(kname, 'SparsePoly')==1
		global SparsePolyKernel;
		global kernel_arg1_degree;
		global kernel_arg2_inhomogene;
		kernel=SparsePolyKernel(size_cache, kernel_arg1_degree, ...
			tobool(kernel_arg2_inhomogene));
		kernel.init(feats_train, feats_train);

	elseif strcmp(kname, 'WeightedCommWordString')==1
		global WeightedCommWordStringKernel;
		global kernel_arg1_use_sign;
		kernel=WeightedCommWordStringKernel(feats_train, feats_train, ...
			tobool(kernel_arg1_use_sign));

	elseif strcmp(kname, 'WeightedDegreePositionString')==1
		global WeightedDegreePositionStringKernel;
		global kernel_arg0_degree;
		global kernel_arg1_degree;
		if isempty(kernel_arg0_degree)
			degree=kernel_arg1_degree;
		else
			degree=kernel_arg0_degree;
		end
		kernel=WeightedDegreePositionStringKernel(...
			feats_train, feats_train, degree);

	elseif strcmp(kname, 'WeightedDegreeString')==1
		global WeightedDegreeStringKernel;
		global kernel_arg0_degree;
		kernel=WeightedDegreeStringKernel(feats_train, feats_train, ...
			kernel_arg0_degree);

	elseif strcmp(kname, 'MatchWordString')==1
		global MatchWordStringKernel;
		global kernel_arg1_degree;
		kernel=MatchWordStringKernel(feats_train, feats_train, ...
			kernel_arg1_degree);

	else
		error('Unsupported kernel %s!', kname);
	end

	y=true;
