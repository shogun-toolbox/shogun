regression <- function(filename) {
	source('util/set_features.R')
	source('util/set_kernel.R')
	source('util/check_accuracy.R')
	source('util/tobool.R')
	source('util/fix_regression_name_inconsistency.R')

	if (!set_features('kernel_')) {
		return(TRUE)
	}

	if (!set_kernel()) {
		return(TRUE);
	}

	sg('threads', regression_num_threads)
	sg('set_labels', 'TRAIN', regression_labels)

	rname <- fix_regression_name_inconsistency(regression_name)
	try(sg('new_regression', rname))

	if (regexpr('svm', regression_type)>0) {
		sg('c', regression_C)
		sg('svm_epsilon', regression_epsilon)
		sg('svr_tube_epsilon', regression_tube_epsilon)
	} else if (regexpr('kernelmachine', regression_type)>0) {
		sg('krr_tau', regression_tau)
	} else {
		print('Incomplete regression data!')
	}

	sg('train_regression')

	alphas <- 0
	bias <- 0
	sv <- 0
	if (exists('regression_bias')) {
		res <- sg('get_svm')

		bias <- abs(res[[1]]-regression_bias)

		weights <- t(res[[2]])
		for (i in 1:length(weights[1,]) ){
			alphas <= alphas + weights[1, i]
		}
		alphas <- abs(alphas-regression_alpha_sum)
		for (i in 1:length(weights[2,])) {
			sv <- sv + weights[2, i]
		}
		sv <- abs(sv-regression_sv_sum)
	}

	classified <- max(abs(sg('classify')-regression_classified))

	data <- list(alphas, bias, sv, classified)
	return(check_accuracy(regression_accuracy, 'classifier', data))
}
