preproc <- function(filename) {
	source('util/set_features.R')
	source('util/set_kernel.R')
	source('util/check_accuracy.R')
	source('util/tobool.R')
	source('util/fix_preproc_name_inconsistency.R');

	if (!set_features('kernel_')) {
		return(TRUE)
	}

	pname <- fix_preproc_name_inconsistency(preproc_name)
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

	kmatrix <- sg('get_kernel_matrix', 'TRAIN')
	km_train <- max(max(abs(kernel_matrix_train-kmatrix)))

	kmatrix <- sg('get_kernel_matrix', 'TEST')
	km_test <- max(max(abs(kernel_matrix_test-kmatrix)))

	data <- list(km_train, km_test)
	return(check_accuracy(kernel_accuracy, 'kernel', data))
}
