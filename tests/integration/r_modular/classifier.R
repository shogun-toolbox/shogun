classifier <- function(filename) {
	source('util/get_features.R')
	source('util/get_kernel.R')
	source('util/get_distance.R')
	source('util/check_accuracy.R')
	source('util/tobool.R')

	if (regexpr('kernel', classifier_type)>0) {
		feats <- get_features('kernel_')
		if (typeof(feats)=='logical') {
			return(TRUE)
		}
		kernel <- get_kernel(feats)
		if (typeof(kernel)=='logical') {
			return(TRUE)
		}
	} else if (regexpr('knn', classifier_type)>0) {
		feats <- get_features('distance_')
		if (typeof(feats)=='logical') {
			return(TRUE)
		}
		distance <- get_distance(feats)
		if (typeof(kernel)=='logical') {
			return(TRUE)
		}
	} else {
		feats <- get_features('classifier_')
		if (typeof(feats)=='logical') {
			return(TRUE)
		}
	}

	if (exists('classifier_labels')) {
		lab <- Labels(as.double(classifier_labels))
	}

	if (regexpr('GMNPSVM', classifier_name)>0) {
		classifier <- GMNPSVM(classifier_C, kernel, lab)
	} else if (regexpr('GPBTSVM', classifier_name)>0) {
		classifier <- GPBTSVM(classifier_C, kernel, lab)
	} else if (regexpr('KNN', classifier_name)>0) {
		classifier <- KNN(as.integer(classifier_k), distance, lab)
	} else if (regexpr('LDA', classifier_name)>0) {
		classifier <- LDA(classifier_gamma, feats[[1]], lab)
	} else if (regexpr('LibLinear', classifier_name)>0) {
		classifier <- LibLinear(classifier_C, feats[[1]], lab)
		classifier$set_solver_type(classifier, L2R_LR)
	} else if (regexpr('LibSVMMultiClass', classifier_name)>0) {
		classifier <- LibSVMMultiClass(classifier_C, kernel, lab)
	} else if (regexpr('LibSVMOneClass', classifier_name)>0) {
		classifier <- LibSVMOneClass(classifier_C, kernel)
	} else if (regexpr('LibSVM', classifier_name)>0) {
		classifier <- LibSVM(classifier_C, kernel, lab)
	} else if (regexpr('MPDSVM', classifier_name)>0) {
		classifier <- MPDSVM(classifier_C, kernel, lab)
	} else if (regexpr('Perceptron', classifier_name)>0) {
		classifier <- Perceptron(feats[[1]], lab)
		classifier$set_learn_rate(classifier, classifier_learn_rate)
		classifier$set_max_iter(classifier, as.integer(classifier_max_iter))
	} else if (regexpr('SVMLight', classifier_name)>0) {
		try(classifier <- SVMLight(classifier_C, kernel, lab))
	} else if (regexpr('SVMLin', classifier_name)>0) {
		classifier <- SVMLin(classifier_C, feats[[1]], lab)
	} else if (regexpr('SVMOcas', classifier_name)>0) {
		classifier <- SVMOcas(classifier_C, feats[[1]], lab)
	} else if (regexpr('SVMSGD', classifier_name)>0) {
		classifier <- SVMSGD(classifier_C, feats[[1]], lab)
	} else if (regexpr('SubGradientSVM', classifier_name)>0) {
		classifier <- SubGradientSVM(classifier_C, feats[[1]], lab)
	} else {
		print(paste('Unsupported classifier', classifier_name))
		return(FALSE)
	}

	classifier$parallel$set_num_threads(
		classifier$parallel, as.integer(classifier_num_threads))

	if (regexpr('linear', classifier_type)>0 && exists('classifier_bias')) {
		classifier$set_bias_enabled(classifier, TRUE)
	}
	if (exists('classifier_epsilon')) {
		if (regexpr('SVMSGD', classifier_name)<=0) {
			classifier$set_epsilon(classifier, classifier_epsilon)
		}
	}
	if (exists('classifier_max_train_time')) {
		classifier$set_max_train_time(classifier, classifier_max_train_time)
	}
	if (exists('classifier_linadd_enabled')) {
		classifier$set_linadd_enabled(
			classifier, tobool(classifier_linadd_enabled))
	}
	if (exists('classifier_batch_enabled')) {
		classifier$set_batch_computation_enabled(
			classifier, tobool(classifier_batch_enabled))
	}

	classifier$train()

	bias <- 0
	if (exists('classifier_bias')) {
		# get_bias yields true instead of a float???
		#print(classifier$get_bias())
		#print(classifier$get_bias(classifier))
		#bias <- classifier$get_bias(classifier)
		#bias <- abs(bias-classifier_bias)
	}

	alphas <- 0
	sv <- 0
	if (exists('classifier_alpha_sum') && exists('classifier_sv_sum')) {
		if (exists('classifier_label_type') && regexpr('series', classifier_label_type)>0) {
			for (i in 0:(classifier$get_num_svms()-1)) {
				subsvm <- classifier$get_svm(classifier, i)
				tmp <- subsvm$get_alphas()
				for (j in 1:length(tmp)) {
					alphas <- alphas + tmp[j]
				}
				tmp <- subsvm$get_support_vectors()
				for (j in 1:length(tmp)) {
					sv <- sv + tmp[j]
				}
			}
			alphas <- abs(alphas-classifier_alpha_sum)
			sv <- abs(sv-classifier_sv_sum)
		} else {
			tmp <- classifier$get_alphas()
			for (i in 1:length(tmp)) {
				alphas <- alphas + tmp[i]
			}
			alphas <- abs(alphas-classifier_alpha_sum)
			tmp <- classifier$get_support_vectors()
			for (i in 1:length(tmp)) {
				sv <- sv + tmp[i]
			}
			sv <- abs(sv-classifier_sv_sum)
		}
	}

	if (regexpr('knn', classifier_type)>0) {
		distance$init(distance, feats[[1]], feats[[2]])
	} else if (regexpr('kernel', classifier_type)>0) {
		kernel$init(kernel, feats[[1]], feats[[2]])
	} else if (regexpr('lda', classifier_type)>0 ||
		regexpr('linear', classifier_type)>0 ||
		regexpr('perceptron', classifier_type)>0) {
		classifier$set_features(classifier, feats[[2]])
	}

	classified <- classifier$classify(classifier)
	classified <- max(abs(
		classified$get_labels(classified)-classifier_classified))

	data <- list(alphas, bias, sv, classified)
	return(check_accuracy(classifier_accuracy, 'classifier', data))
}
