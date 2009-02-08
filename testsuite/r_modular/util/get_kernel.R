get_kernel <- function(feats) {
	source('util/tobool.R')
	source('util/get_distance.R')
	source('util/get_pie.R')

	if (exists('kernel_arg0_size')) {
		size_cache <- as.integer(kernel_arg0_size)
	}
	else if (exists('kernel_arg1_size')) {
		size_cache <- as.integer(kernel_arg1_size)
	} else {
		size_cache <- as.integer(10)
	}

	if (exists('kernel_name')) {
		kname <- kernel_name
	} else {
		kname <- name
	}

	if (regexpr('^AUC', kname)>0) {
		if (regexpr('^Gaussian', kernel_subkernel0_name)==0) {
			print(paste(
				'Dunno how to handle AUC subkernel', kernel_subkernel0_name))
		}
		subfeats_train <- RealFeatures(kernel_subkernel0_data_train)
		subfeats_test <- RealFeatures(kernel_subkernel0_data_test)
		subkernel <- GaussianKernel(
			subfeats_train, subfeats_test, kernel_subkernel0_arg1_width)
		return(
			AUCKernel(feats[[1]], feats[[1]], kernel_arg1_width, size_cache))
	}

	else if (regexpr('^Chi2', kname)>0) {
		return(Chi2Kernel(
			feats[[1]], feats[[1]], kernel_arg1_width, size_cache))
	}

	else if (regexpr('^Combined', kname)>0) {
		# this will break when structure of test file is changed!
		kernel <- CombinedKernel()
		subkernel <- FixedDegreeStringKernel(
			as.integer(kernel_subkernel0_arg0_size),
			as.integer(kernel_subkernel0_arg1_degree))
		kernel$append_kernel(kernel, subkernel)
		subkernel <- PolyMatchStringKernel(
			as.integer(kernel_subkernel1_arg0_size),
			as.integer(kernel_subkernel1_arg1_degree),
			tobool(kernel_subkernel1_arg2_inhomogene))
		kernel$append_kernel(kernel, subkernel)
		subkernel <- LocalAlignmentStringKernel(
			as.integer(kernel_subkernel2_arg0_size))
		kernel$append_kernel(kernel, subkernel)
		kernel$init(kernel, feats[[1]], feats[[1]])
		return(kernel)
	}

	else if (regexpr('^CommUlongString', kname)>0) {
		if (exists('kernel_arg1_use_sign')) {
			return(CommUlongStringKernel(feats[[1]], feats[[1]],
				tobool(kernel_arg1_use_sign)))
		} else {
			return(CommUlongStringKernel(feats[[1]], feats[[1]]))
		}
	}

	else if (regexpr('^CommWordString', kname)>0) {
		if (exists('kernel_arg1_use_sign')) {
			return(CommWordStringKernel(feats[[1]], feats[[1]],
				tobool(kernel_arg1_use_sign)))
		} else {
			return(CommWordStringKernel(feats[[1]], feats[[1]]))
		}
	}

	else if (regexpr('^Const', kname)>0) {
		return(ConstKernel(feats[[1]], feats[[1]], kernel_arg0_c))
	}

	else if (regexpr('^Custom', kname)>0) {
		return(TRUE) # silently unsupported
	}

	else if (regexpr('^Diag', kname)>0) {
		return(DiagKernel(feats[[1]], feats[[1]], kernel_arg1_diag))
	}

	else if (regexpr('^Distance', kname)>0) {
		distance <- get_distance(feats)
		if (typeof(distance)=='logical') {
			return(distance)
		}
		return(DistanceKernel(
			feats[[1]], feats[[1]], kernel_arg1_width, distance))
	}

	else if (regexpr('^FixedDegreeString', kname)>0) {
		return(FixedDegreeStringKernel(
			feats[[1]], feats[[1]], as.integer(kernel_arg1_degree)))
	}

	else if (regexpr('^Gaussian$', kname)>0) {
		if (exists('kernel_arg0_width')) {
			width <- kernel_arg0_width
		} else {
			width <- kernel_arg1_width
		}
		return(GaussianKernel(feats[[1]], feats[[1]], width))
	}

	else if (regexpr('^GaussianShift', kname)>0) {
		return(GaussianShiftKernel(
			feats[[1]], feats[[1]],
			kernel_arg1_width,
			as.integer(kernel_arg2_max_shift),
			as.integer(kernel_arg3_shift_step)))
	}

	else if (regexpr('^HistogramWordString', kname)>0) {
		pie <- get_pie(feats[[1]])
		if (typeof(pie)=='logical') {
			return(TRUE)
		}
		return(HistogramWordStringKernel(feats[[1]], feats[[1]], pie))
	}

	else if (regexpr('^Linear$', kname)>0) {
		kernel <- LinearKernel()
		if (exists('kernel_arg0_scale')) {
			normalizer <- AvgDiagKernelNormalizer(kernel_arg0_scale)
			kernel$set_normalizer(kernel, normalizer)
		} else {
			kernel$set_normalizer(kernel, AvgDiagKernelNormalizer())
		}
		kernel$init(kernel, feats[[1]], feats[[1]])
		return(kernel)
	}

	else if (regexpr('^LinearByte', kname)>0) {
		kernel <- LinearByteKernel()
		kernel$set_normalizer(kernel, AvgDiagKernelNormalizer())
		kernel$init(kernel, feats[[1]], feats[[1]])
		return(kernel)
	}

	else if (regexpr('^LinearString', kname)>0) {
		kernel <- LinearStringKernel()
		kernel$set_normalizer(kernel, AvgDiagKernelNormalizer())
		kernel$init(kernel, feats[[1]], feats[[1]])
		return(kernel)
	}

	else if (regexpr('^LinearWord', kname)>0) {
		kernel <- LinearWordKernel()
		kernel$set_normalizer(kernel, AvgDiagKernelNormalizer())
		kernel$init(kernel, feats[[1]], feats[[1]])
		return(kernel)
	}

	else if (regexpr('^LocalAlignmentString', kname)>0) {
		return(LocalAlignmentStringKernel(feats[[1]], feats[[1]]))
	}

	else if (regexpr('^MatchWordString', kname)>0) {
		return(MatchWordStringKernel(
			feats[[1]], feats[[1]], as.integer(kernel_arg1_degree)))
	}

	else if (regexpr('^Oligo', kname)>0) {
		kernel <- OligoKernel(size_cache,
			as.integer(kernel_arg1_k), kernel_arg2_width)
		kernel$init(kernel, feats[[1]], feats[[1]])
		return(kernel)
	}


	else if (regexpr('^PolyMatchString', kname)>0) {
		return(PolyMatchStringKernel(
			feats[[1]], feats[[1]],
			as.integer(kernel_arg1_degree), tobool(kernel_arg2_inhomogene)))
	}

	else if (regexpr('^PolyMatchWordString', kname)>0) {
		return(PolyMatchWordStringKernel(feats[[1]], feats[[1]],
			as.integer(kernel_arg1_degree), tobool(kernel_arg2_inhomogene)))
	}

	else if (regexpr('^Poly', kname)>0) {
		if (exists('kernel_normalizer')) {
			kernel <- PolyKernel(size_cache,
				as.integer(kernel_arg1_degree),
				tobool(kernel_arg2_inhomogene))
			normalizer <- eval(parse(
				text=paste(kernel_normalizer, '()', sep='')))
			kernel$set_normalizer(kernel, normalizer)
			kernel$init(kernel, feats[[1]], feats[[1]])
		} else {
			kernel <- PolyKernel(
				feats[[1]], feats[[1]],
				as.integer(kernel_arg1_degree), tobool(kernel_arg2_inhomogene))
		}
		return(kernel)
	}

	else if (regexpr('^SalzbergWordString', kname)>0) {
		pie <- get_pie(feats[[1]])
		if (typeof(pie)=='logical') {
			return(TRUE)
		}
		return(SalzbergWordStringKernel(feats[[1]], feats[[1]], pie))
	}

	else if (regexpr('^Sigmoid', kname)>0) {
		return(SigmoidKernel(
			feats[[1]], feats[[1]],
			size_cache, kernel_arg1_gamma, kernel_arg2_coef0))
	}

	else if (regexpr('^SimpleLocalityImprovedString', kname)>0) {
		return(SimpleLocalityImprovedStringKernel(
			feats[[1]], feats[[1]],
			as.integer(kernel_arg1_length),
			as.integer(kernel_arg2_inner_degree),
			as.integer(kernel_arg3_outer_degree)))
	}

	else if (regexpr('^SparseGaussian', kname)>0) {
		return(SparseGaussianKernel(feats[[1]], feats[[1]], kernel_arg1_width))
	}

	else if (regexpr('^SparseLinear', kname)>0) {
		kernel <- SparseLinearKernel()
		if (exists('kernel_arg0_scale')) {
			normalizer <- AvgDiagKernelNormalizer(kernel_arg0_scale)
			kernel$set_normalizer(kernel, normalizer)
		} else {
			kernel$set_normalizer(kernel, AvgDiagKernelNormalizer())
		}
		kernel$init(kernel, feats[[1]], feats[[1]])
		return(kernel)
	}

	else if (regexpr('^SparsePoly', kname)>0) {
		kernel <- SparsePolyKernel(
			size_cache, as.integer(kernel_arg1_degree),
			tobool(kernel_arg2_inhomogene))
		kernel$init(kernel, feats[[1]], feats[[1]])
		return(kernel)
	}

	else if (regexpr('^WeightedCommWordString', kname)>0) {
		return(WeightedCommWordStringKernel(
			feats[[1]], feats[[1]], tobool(kernel_arg1_use_sign)))
	}

	else if (regexpr('^WeightedDegreePositionString', kname)>0) {
		if (exists('kernel_arg0_degree')) {
			degree <- as.integer(kernel_arg0_degree)
		} else {
			degree <- as.integer(kernel_arg1_degree)
		}
		return(WeightedDegreePositionStringKernel(
			feats[[1]], feats[[1]], degree))
	}

	else if (regexpr('^WeightedDegreeString', kname)>0) {
		return(WeightedDegreeStringKernel(
			feats[[1]], feats[[1]], as.integer(kernel_arg0_degree)))
	}

	else {
		print(paste('Unsupported kernel:', kname))
		return(TRUE)
	}
}
