set_kernel <- function() {
	source('util/tobool.R')
	source('util/fix_kernel_name_inconsistency.R')

	if (exists('kernel_name')) {
		kname <- fix_kernel_name_inconsistency(kernel_name)
	} else {
		kname <- fix_kernel_name_inconsistency(name)
	}

	if (regexpr(kname, 'AUC')>0 || regexpr(kname, 'CUSTOM')>0) {
		print(paste('Kernel', kname, 'not supported yet!'))
		return(FALSE)
	}

	if (exists('kernel_arg0_size')) {
		size_cache <- kernel_arg0_size
	}
	else if (exists('kernel_arg1_size')) {
		size_cache <- kernel_arg1_size
	} else {
		size_cache <- 10
	}

	ftype <- toupper(feature_type)

	if (regexpr('SIGMOID', kname)>0) {
		sg('set_kernel', kname, ftype, size_cache,
			kernel_arg1_gamma, kernel_arg2_coef0)
	}

	else if (regexpr('CHI2', kname)>0) {
		sg('set_kernel', kname, ftype, size_cache, kernel_arg0_width)
	}

	else if (regexpr('CONST', kname)>0) {
		sg('set_kernel', kname, ftype, size_cache, kernel_arg0_c)
	}

	else if (regexpr('DIAG', kname)>0) {
		sg('set_kernel', kname, ftype, size_cache, kernel_arg0_diag)
	}

	else if (regexpr('GAUSSIANSHIFT', kname)>0) {
		sg('set_kernel', kname, ftype, size_cache,
			kernel_arg0_width, kernel_arg1_max_shift, kernel_arg2_shift_step)
	}

	else if (regexpr('GAUSSIAN', kname)>0) {
		sg('set_kernel', kname, ftype, size_cache, kernel_arg0_width)
	}

	else if (regexpr('LINEAR', kname)>0) {
		if (exists('kernel_arg0_scale')) {
			sg('set_kernel', kname, ftype, size_cache, kernel_arg0_scale)
		} else {
			sg('set_kernel', kname, ftype, size_cache, -1)
		}
	}

	else if (regexpr('POLYMATCH', kname)>0) {
		sg('set_kernel', kname, ftype, size_cache,
			kernel_arg0_degree, tobool(kernel_arg1_inhomogene))
	}

	else if (regexpr('POLY', kname)>0) {
		sg('set_kernel', kname, ftype, size_cache,
			kernel_arg0_degree, tobool(kernel_arg1_inhomogene),
			tobool(kernel_arg2_use_normalization))
	}

	else if (regexpr('MATCH', kname)>0) {
		sg('set_kernel', kname, ftype, size_cache, kernel_arg0_degree)
	}

	else if (regexpr('COMMSTRING', kname)>0) { # normal + WEIGHTED
		source('util/fix_normalization_inconsistency.R')
		norm <- fix_normalization_inconsistency(kernel_arg1_normalization)
		sg('set_kernel', kname, ftype, size_cache,
			tobool(kernel_arg0_use_sign), norm)
	}

	else if (regexpr('DEGREE', kname)>0) { # FIXED + WEIGHTED
		sg('set_kernel', kname, ftype, size_cache, kernel_arg0_degree)
	}

	else if (regexpr('HISTOGRAM', kname)>0 || regexpr('SALZBERG', kname)>0) {
		pseudo_pos=1e-10
		pseudo_neg=1e-10
		sg('new_plugin_estimator', pseudo_pos, pseudo_neg)
		sg('set_labels', 'TRAIN', classifier_labels)
		sg('train_estimator')

		sg('set_kernel', kname, ftype, size_cache)
	}

	else if (regexpr('DISTANCE', kname)>0) {
		source('util/fix_distance_name_inconsistency.R')
		dname <- fix_distance_name_inconsistency(kernel_arg1_distance)
		# FIXME: REAL is cheating and will break in the future
		sg('set_distance', dname, 'REAL')
		sg('set_kernel', kname, size_cache, kernel_arg0_width)
	}

	else if (regexpr('LOCALALIGNMENT', kname)>0) {
		sg('set_kernel', kname, ftype, size_cache)
	}

	else if (regexpr('LIK', kname)>0) {
		sg('set_kernel', kname, ftype, size_cache,
			kernel_arg0_length, kernel_arg1_inner_degree,
			kernel_arg2_outer_degree)
	}

	else if (regexpr('COMBINED', kname)>0) {
		# this will break when test file is changed!
		sg('set_kernel', 'COMBINED', size_cache)

		subkernel_name=fix_kernel_name_inconsistency(subkernel0_name)
		sg('add_kernel', 1., subkernel_name,
			toupper(subkernel0_feature_type),
			subkernel0_kernel_arg0_size,
			subkernel0_kernel_arg1_degree)

		subkernel_name=fix_kernel_name_inconsistency(subkernel1_name)
		sg('add_kernel', 1., subkernel_name,
			toupper(subkernel1_feature_type),
			subkernel1_kernel_arg0_size,
			subkernel1_kernel_arg1_degree,
			tobool(subkernel1_kernel_arg2_inhomogene))

		subkernel_name=fix_kernel_name_inconsistency(subkernel2_name)
		sg('add_kernel', 1., subkernel_name,
			toupper(subkernel2_feature_type),
			subkernel2_kernel_arg0_size)
	}

	else {
		return(FALSE)
	}

	sg('init_kernel', 'TRAIN')
	return(TRUE)
}
