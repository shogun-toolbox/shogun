kernel <- function() {
	source('util/set_features.R')
	source('util/set_kernel.R')
	source('util/check_accuracy.R')

	if (!set_features()) {
		return(TRUE)
	}

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
