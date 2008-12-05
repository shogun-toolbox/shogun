kernel <- function() {
	source('util/get_features.R')
	source('util/get_kernel.R')
	source('util/check_accuracy.R')

	feats <- get_features('kernel_')
	if (typeof(feats)=='logical') {
		return(TRUE)
	}

	kernel <- get_kernel(feats)
	if (typeof(kernel)=='logical') {
		return(TRUE)
	}

	kmatrix <- kernel$get_kernel_matrix()
	km_train <- max(max(abs(kernel_matrix_train-kmatrix)))

	kernel$init(kernel, feats[[1]], feats[[2]])
	kmatrix <- kernel$get_kernel_matrix()
	km_test <- max(max(abs(kernel_matrix_test-kmatrix)))
	data <- list(km_train, km_test)
	return(check_accuracy(kernel_accuracy, 'kernel', data))
}
