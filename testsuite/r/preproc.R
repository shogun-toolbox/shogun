preproc <- function(filename) {
	source('util/set_features.R')
	source('util/set_kernel.R')
	source('util/check_accuracy.R')
	source('util/tobool.R')
	source('util/fix_preproc_name_inconsistency.R');

	if (!set_features()) {
		return(TRUE)
	}

	pname <- fix_preproc_name_inconsistency(name)
	if (regexpr('PRUNEVARSUBMEAN', pname)>0) {
		sg('add_preproc', pname, tobool(preproc_arg0_divide))
	} else {
		sg('add_preproc', pname)
	}

	sg('attach_preproc', 'TRAIN')
	sg('attach_preproc', 'TEST')

	if (!set_kernel()) {
		return(TRUE);
	}

	kmatrix <- sg('get_kernel_matrix')
	ktrain <- max(max(abs(km_train-kmatrix)))

	sg('init_kernel', 'TEST');
	kmatrix <- sg('get_kernel_matrix')
	ktest <- max(max(abs(km_test-kmatrix)))

	data <- list(ktrain, ktest)
	return(check_accuracy(accuracy, 'kernel', data))
}
