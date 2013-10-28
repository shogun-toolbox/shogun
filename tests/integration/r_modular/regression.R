regression <- function(filename) {
	source('util/get_features.R')
	source('util/get_kernel.R')
	source('util/check_accuracy.R')
	source('util/tobool.R')

	feats <- get_features('kernel_')
	if (typeof(feats)=='logical') {
		return(TRUE)
	}
	kernel <- get_kernel(feats)
	if (typeof(kernel)=='logical') {
		return(TRUE)
	}

	regression_num_threads <- as.integer(regression_num_threads)
	kernel$parallel$set_num_threads(kernel$parallel, regression_num_threads)
	lab <- Labels(as.double(regression_labels))

	if (regexpr('KRR', regression_name)>0) {
		regression <- KRR(regression_tau, kernel, lab)
	} else if (regexpr('LibSVR', regression_name)>0) {
		regression <- LibSVR(regression_C, regression_epsilon, kernel, lab)
		regression$set_tube_epsilon(regression, regression_tube_epsilon)
	} else if (regexpr('SVRLight', regression_name)) {
		try(regression <- SVRLight(
			regression_C, regression_epsilon, kernel, lab))
		regression$set_tube_epsilon(regression, regression_tube_epsilon)
	} else {
		print(paste('Unsupported regression', regression_name))
		return(FALSE)
	}

	regression$parallel$set_num_threads(
		regression$parallel, regression_num_threads)
	regression$train(regression)

	bias <- 0
	if (exists('regression_bias')) {
		bias <- abs(regression$get_bias()-regression_bias)
	}

	alphas <- 0
	sv <- 0
	if (exists('regression_alpha_sum')) {
		tmp <- regression$get_alphas()
		for (i in 1:length(tmp) ){
			alphas <= alphas + tmp[i]
		}
		alphas <- abs(alphas-regression_alpha_sum)

		tmp <- regression$get_support_vectors()
		for (i in 1:length(tmp)) {
			sv <- sv + tmp[i]
		}
		sv <- abs(sv-regression_sv_sum)
	}

	kernel$init(kernel, feats[[1]], feats[[2]])
	classified <- regression$classify(regression)
	classified <- max(abs(
		classified$get_labels(classified)-regression_classified))

	data <- list(alphas, bias, sv, classified)
	return(check_accuracy(regression_accuracy, 'classifier', data))
}
