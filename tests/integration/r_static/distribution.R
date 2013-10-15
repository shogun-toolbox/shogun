distribution <- function(filename) {
	source('util/set_features.R')
	source('util/check_accuracy.R')

	sg('init_random', init_random)
	set.seed(init_random)

	if (!set_features('distribution_')) {
		return(TRUE)
	}

	if (regexpr('^HMM', distribution_name)>0) {
		sg('new_hmm', distribution_N, distribution_M)
		sg('bw')
	} else {
		print('Cannot yet train other distributions than HMM!')
		return(TRUE)
	}

	likelihood <- abs(sg('hmm_likelihood')-distribution_likelihood)
	data <- list(likelihood, 0)
	return(check_accuracy(distribution_accuracy, 'distribution', data))
}
