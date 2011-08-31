classifier <- function(filename) {
	source('util/set_features.R')
	source('util/set_kernel.R')
	source('util/set_distance.R')
	source('util/check_accuracy.R')
	source('util/fix_classifier_name_inconsistency.R')

	if (regexpr('Perceptron', classifier_name)>0) { # b0rked, skip it
		return(TRUE)
	}

	if (regexpr('kernel', classifier_type)>0) {
		if (!set_features('kernel_')) {
			return(TRUE)
		}
		if (!set_kernel()) {
			return(TRUE)
		}
	} else if (regexpr('knn', classifier_type)>0) {
		if (!set_features('distance_')) {
			return(TRUE)
		}
		if (!set_distance()) {
			return(TRUE)
		}
	} else {
		if (!set_features('classifier_')) {
			return(TRUE)
		}
	}

	if (exists('classifier_labels')) {
		sg('set_labels', 'TRAIN', classifier_labels)
	}

	cname <- fix_classifier_name_inconsistency(classifier_name)
	try(sg('new_classifier', cname))

	if (exists('classifier_bias')) {
		sg('svm_use_bias', TRUE)
	} else {
		sg('svm_use_bias', FALSE)
	}

	if (exists('classifier_epsilon')) {
		sg('svm_epsilon', classifier_epsilon)
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

	if (regexpr('lda', classifier_type)>0) {
		0 # nop
	} else {
		if (exists('classifier_bias') && exists('classifier_label_type') && regexpr('series', classifier_label_type)<=0) {
			res <- sg('get_svm')
			bias <- abs(res[[1]]-classifier_bias)
		}

		if (exists('classifier_alpha_sum') && exists('classifier_sv_sum')) {
			if (exists('classifier_label_type') && regexpr('series', classifier_label_type)>0) {
				for (i in 0:(sg('get_num_svms')-1)) {
					weights <- t(sg('get_svm', i)[[2]])
					for (j in 1:length(weights[1,])) {
						alphas <- alphas + weights[1, j]
					}
					for (j in 1:length(weights[2,])) {
						sv <- sv + weights[2, j]
					}
				}
				alphas <- abs(alphas-classifier_alpha_sum)
				sv <- abs(sv-classifier_sv_sum)
			} else {
				weights <- t(sg('get_svm')[[2]])
				for (i in 1:length(weights[1,]) ){
					alphas <- alphas + weights[1, i]
				}
				alphas <- abs(alphas-classifier_alpha_sum)
				for (i in 1:length(weights[2,])) {
					sv <- sv + weights[2, i]
				}
				sv <- abs(sv-classifier_sv_sum)
			}
		}
	}

	classified <- max(abs(sg('classify')-classifier_classified))

	data <- list(alphas, bias, sv, classified)
	return(check_accuracy(classifier_accuracy, 'classifier', data))
}
