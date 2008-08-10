classifier <- function(filename) {
	source('util/set_features.R')
	source('util/set_kernel.R')
	source('util/set_distance.R')
	source('util/check_accuracy.R')
	source('util/fix_classifier_name_inconsistency.R')

	if (regexpr('Perceptron', name)>0) { # b0rked, skip it
		return(TRUE)
	}

	if (!set_features()) {
		return(TRUE)
	}

	if (regexpr('kernel', classifier_type)>0) {
		if (!set_kernel()) {
			return(TRUE)
		}
	} else if (regexpr('knn', classifier_type)>0) {
		if (!set_distance()) {
			return(TRUE)
		}
	}

	if (exists('classifier_labels')) {
		sg('set_labels', 'TRAIN', classifier_labels)
	}

	cname <- fix_classifier_name_inconsistency(name)
	try(sg('new_classifier', cname))

	if (exists('classifier_bias')) {
		sg('svm_use_bias', TRUE)
	} else {
		sg('svm_use_bias', FALSE)
	}

	if (exists('classifier_epsilon')) {
		sg('svm_epsilon', classifier_epsilon)
	}
	if (exists('classifier_tube_epsilon')) {
		sg('svr_tube_epsilon', classifier_tube_epsilon)
	}
	if (exists('classifier_max_train_time')) {
		sg('svm_max_train_time', classifier_max_train_time)
	}
	if (exists('classifier_linadd_enabled')) {
		sg('use_linadd', TRUE)
	}
	if (exists('classifier_batch_enabled')) {
		sg('use_batch_computation', TRUE)
	}
	if (exists('classifier_num_threads')) {
		sg('threads', classifier_num_threads)
	}

	if (regexpr('knn', classifier_type)>0) {
		sg('train_classifier', classifier_k)
	} else if (regexpr('lda', classifier_type)>0) {
		sg('train_classifier', classifier_gamma)
	} else {
		if (exists('classifier_C')) {
			sg('c', classifier_C)
		}
		sg('train_classifier')
	}

	alphas <- 0
	bias <- 0
	sv <- 0

	if (regexpr('knn', classifier_type)>0) {
		sg('init_distance', 'TEST')
	} else if (regexpr('lda', classifier_type)>0) {
		0 # nop
	} else {
		if (exists('regression_bias')) {
			res <- sg('get_svm')

			bias <- abs(res[[1]]-regression_bias)

			weights <- t(res[[2]])
			alphas <- max(abs(weights[1,]-regression_alphas))
			sv <- max(abs(weights[2,]-regression_support_vectors))
		}

		sg('init_kernel', 'TEST')
	}

	classified <- max(abs(sg('classify')-classifier_classified))

	data <- list(alphas, bias, sv, classified)
	return(check_accuracy(classifier_accuracy, 'classifier', data))
}
