get_features <- function(prefix) {
	source('util/convert_features_and_add_preproc.R')
	source('util/get_posneg.R')

	size_cache <- 10
	name <- eval(parse(text=paste(prefix, 'name', sep='')))

	if (regexpr('Combined', name)>0) {
		# this will break when subkernels in data file are changed
		# it blindly assumes that all features are StringChar
		train <- CombinedFeatures()
		test <- CombinedFeatures()

		feat <- StringCharFeatures('DNA')
		feat$set_string_features(feat, kernel_subkernel0_data_train)
		train$append_feature_obj(train, feat)
		feat <- StringCharFeatures('DNA')
		feat$set_string_features(feat, kernel_subkernel0_data_test)
		test$append_feature_obj(test, feat)

		feat <- StringCharFeatures('DNA')
		feat$set_string_features(feat, kernel_subkernel1_data_train)
		train$append_feature_obj(train, feat)
		feat <- StringCharFeatures('DNA')
		feat$set_string_features(feat, kernel_subkernel1_data_test)
		test$append_feature_obj(test, feat)

		feat <- StringCharFeatures('DNA')
		feat$set_string_features(feat, kernel_subkernel2_data_train)
		train$append_feature_obj(train, feat)
		feat <- StringCharFeatures('DNA')
		feat$set_string_features(feat, kernel_subkernel2_data_test)
		test$append_feature_obj(test, feat)
	} else if (regexpr('Custom', name)>0) {
		train <- RealFeatures(kernel_data)
		test <- RealFeatures(kernel_data)
	} else if (exists('topfk_name')) {
		posneg <- get_posneg('topfk_')
		if (typeof(posneg)=='logical') {
			return(posneg)
		}

		if (regexpr('FK', topfk_name)>0) {
			train <- FKFeatures(size_cache, posneg[1][1], posneg[1][2])
			train$set_opt_a(-1) # estimate prior
			test <- FKFeatures(size_cache, posneg[2][1], posneg[2][2])
			test$set_a(train$get_a())
		} else {
			train <- TOPFeatures(size_cache, posneg[1][1], posneg[1][2])
			test <- TOPFeatures(size_cache, posneg[2][1], posneg[2][2])
		}
	} else {
		data_train <- eval(parse(text=paste(prefix, 'data_train', sep='')))
		data_test <- eval(parse(text=paste(prefix, 'data_test', sep='')))
		feature_type <- eval(parse(text=paste(prefix, 'feature_type', sep='')))
		feature_class <- eval(parse(
			text=paste(prefix, 'feature_class', sep='')))

		if (regexpr('Sparse', name)>0) {
			is_sparse <- TRUE
		} else if (exists('classifier_type') &&
			regexpr('linear', classifier_type)>0) {
			is_sparse <- TRUE
		} else {
			is_sparse <- FALSE
		}

		if (regexpr('simple', feature_class)>0) {
			if (regexpr('Real', feature_type)>0) {
				train <- RealFeatures(data_train)
				test <- RealFeatures(data_test)

				if (is_sparse) {
					sparse <- SparseRealFeatures()
					sparse$obtain_from_simple(sparse, train)
					train <- sparse
					sparse <- SparseRealFeatures()
					sparse$obtain_from_simple(sparse, test)
					test <- sparse
				}
			} else if (regexpr('Word', feature_type)) {
				print('Datatype Word not yet supported in R')
				return(FALSE)
				train <- WordFeatures(data_train)
				test <- WordFeatures(data_test)
			} else if (regexpr('Byte', feature_type)) {
				print('Datatype Byte not yet supported in R')
				return(FALSE)
				train <- ByteFeatures(RAWBYTE)
				train$copy_feature_matrix(data_train)
				test <- ByteFeatures(RAWBYTE)
				test$copy_feature_matrix(data_test)
			} else {
				print(paste('Simple feature type not supported yet:',
					feature_type))
			}
		} else if (regexpr('string', feature_class)>0) {
			alphabet <- eval(parse(text=paste(prefix, 'alphabet', sep='')))
			if (grep('DNA', alphabet)==0 && grep('CUBE', alphabet)==0) {
				print(paste('Alphabet not supported yet:', alphabet))
				return(FALSE)
			}

			train <- StringCharFeatures(alphabet)
			train$set_string_features(train, data_train)
			test <- StringCharFeatures(alphabet)
			test$set_string_features(test, data_test)

			if (regexpr('string_complex', feature_class)>0) {
				return(convert_features_and_add_preproc(train, test, prefix))
			}
		} else if (regexpr('wd', feature_class)>0) {
			order <- as.integer(eval(parse(text=paste(
				prefix, 'order', sep=''))))

			charfeat <- StringCharFeatures('DNA')
			charfeat$set_string_features(charfeat, data_train)
			bytefeat <- StringByteFeatures('RAWDNA')
			bytefeat$obtain_from_char(bytefeat, charfeat,
				as.integer(0), as.integer(1), as.integer(0), FALSE)
			train <- WDFeatures(bytefeat, order, order)

			charfeat <- StringCharFeatures('DNA')
			charfeat$set_string_features(charfeat, data_test)
			bytefeat <- StringByteFeatures('RAWDNA')
			bytefeat$obtain_from_char(bytefeat, charfeat,
				as.integer(0), as.integer(1), as.integer(0), FALSE)
			test <- WDFeatures(bytefeat, order, order)
		}
		else {
			print(paste('Unknown feature class', feature_class))
			return(FALSE)
		}
	}

	return(list(train, test))
}
