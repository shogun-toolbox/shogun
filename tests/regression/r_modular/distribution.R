distribution <- function(filename) {
	source('util/get_features.R')
	source('util/check_accuracy.R')

	feats <- get_features('distribution_')
	if (typeof(feats)=='logical') {
		return(TRUE)
	}

	if (regexpr('^HMM', distribution_name)>0) {
		distribution <- HMM(feats[[1]],
			as.integer(distribution_N), as.integer(distribution_M),
			distribution_pseudo)
		distribution$train(distribution)
		distribution$baum_welch_viterbi_train(distribution, 'BW_NORMAL')
	}
	else if (regexpr('^Histogram', distribution_name)>0) {
		distribution <- Histogram(feats[[1]])
		distribution$train(distribution)
	} else if (regexpr('^LinearHMM', distribution_name)>0) {
		distribution <- LinearHMM(feats[[1]])
		distribution$train(distribution)
	} else {
		print(paste('Unknown distribution', distribution_name))
		return(TRUE)
	}

	likelihood <- max(abs(
		distribution$get_log_likelihood_sample()-distribution_likelihood))
	num_examples <- feats[[1]]$get_num_vectors()
	num_param <- distribution$get_num_model_parameters()
	derivatives <- 0
	for (i in 0:(num_param-1)) {
		for (j in 0:(num_examples-1)) {
			val <- distribution$get_log_derivative(
				distribution, as.integer(i), as.integer(j))
			if (abs(val)!=Inf) {
				derivatives <- derivatives + val
			}
		}
	}
	derivatives <- max(abs(derivatives-distribution_derivatives))

	data=list(likelihood, derivatives)
	return(check_accuracy(distribution_accuracy, 'distribution', data))
}
