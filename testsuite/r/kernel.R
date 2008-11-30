kernel <- function() {
	source('util/set_features.R')
	source('util/set_kernel.R')
	source('util/check_accuracy.R')

	if (!set_features('kernel_')) {
		return(TRUE)
	}

	if (!set_kernel()) {
		return(TRUE)
	}

	kmatrix <- sg('get_kernel_matrix')
	km_train <- max(max(abs(kernel_matrix_train-kmatrix)))

	sg('init_kernel', 'TEST')
	kmatrix <- sg('get_kernel_matrix')
	km_test <- max(max(abs(kernel_matrix_test-kmatrix)))

	data <- list(km_train, km_test)
	return(check_accuracy(kernel_accuracy, 'kernel', data))
}

# vim: set filetype=R
