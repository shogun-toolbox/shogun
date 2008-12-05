preproc <- function(filename) {
	source('util/get_features.R')
	source('util/get_kernel.R')
	source('util/check_accuracy.R')
	source('util/tobool.R')

	feats <- get_features('kernel_')
	if (typeof(feats)=='logical') {
		return(TRUE)
	}

	if (regexpr('LogPlusOne', preproc_name)>0) {
		preproc <- LogPlusOne()
	} else if (regexpr('NormOne', preproc_name)>0) {
		preproc <- NormOne()
	} else if (regexpr('PruneVarSubMean', preproc_name)>0) {
		preproc <- PruneVarSubMean(tobool(preproc_arg0_divide))
	} else if (regexpr('SortUlongString', preproc_name)>0) {
		preproc <- SortUlongString()
	} else if (regexpr('SortWordString', preproc_name)>0) {
		preproc <- SortWordString()
	} else {
		print(paste('Unsupported preproc', preproc_name))
		return(FALSE)
	}

	preproc$init(preproc, feats[[1]])
	feats[[1]]$add_preproc(feats[[1]], preproc)
	feats[[1]]$apply_preproc(feats[[1]])
	feats[[2]]$add_preproc(feats[[2]], preproc)
	feats[[2]]$apply_preproc(feats[[2]])

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
