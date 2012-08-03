set_features <- function(prefix) {
	source('util/convert_features_and_add_preproc.R')

	name <- eval(parse(text=paste(prefix, 'name', sep='')))
	if (exists(paste(prefix, 'alphabet', sep=''))) {
		alphabet <- eval(parse(text=paste(prefix, 'alphabet', sep='')))
	}
	if (exists(paste(prefix, 'data_train', sep=''))) {
		data_train <- eval(parse(text=paste(prefix, 'data_train', sep='')))
		data_test <- eval(parse(text=paste(prefix, 'data_test', sep='')))
	}

	if (regexpr('Sparse', name)>0) {
		print('Sparse features not supported yet!')
		return(FALSE)
	}

	if (exists('classifier_type')) {
		if (regexpr('linear', classifier_type)>0) {
			print('Linear classifiers with sparse features not supported yet!')
			return(FALSE)
		}
	}

	if (exists('alphabet')) {
		if (regexpr('RAWBYTE', alphabet)>0) {
			print('Alphabet RAWBYTE not supported yet!')
			return(FALSE)
		}
	}

	if (exists('topfk_name')) {
		print(paste('Features', topfk_name, 'not supported yet!'))
		return(FALSE)
	}

	sg('clean_features', 'TRAIN')
	sg('clean_features', 'TEST')

	if (regexpr('Combined', name)>0) {
		if (exists('kernel_subkernel0_alphabet')) {
			sg('add_features', 'TRAIN',
				kernel_subkernel0_data_train, kernel_subkernel0_alphabet)
			sg('add_features', 'TEST',
				kernel_subkernel0_data_test, kernel_subkernel0_alphabet)
		} else {
			sg('add_features', 'TRAIN', kernel_subkernel0_data_train)
			sg('add_features', 'TEST', kernel_subkernel0_data_test)
		}

		if (exists('kernel_subkernel1_alphabet')) {
			sg('add_features', 'TRAIN',
				kernel_subkernel1_data_train, kernel_subkernel1_alphabet)
			sg('add_features', 'TEST',
				kernel_subkernel1_data_test, kernel_subkernel1_alphabet)
		} else {
			sg('add_features', 'TRAIN', kernel_subkernel1_data_train)
			sg('add_features', 'TEST', kernel_subkernel1_data_test)
		}

		if (exists('kernel_subkernel2_alphabet')) {
			sg('add_features', 'TRAIN',
				kernel_subkernel2_data_train, kernel_subkernel2_alphabet)
			sg('add_features', 'TEST',
				kernel_subkernel2_data_test, kernel_subkernel2_alphabet)
		} else {
			sg('add_features', 'TRAIN', kernel_subkernel2_data_train)
			sg('add_features', 'TEST', kernel_subkernel2_data_test)
		}
	}

	else if (exists('alphabet')) {
		sg('set_features', 'TRAIN', data_train, alphabet)
		sg('set_features', 'TEST', data_test, alphabet)
	}

	else if (exists('data_train')) {
		ftype <- eval(parse(text=paste(prefix, 'feature_type', sep='')))
		if (regexpr('Real', ftype)<0) {
			print(paste('No support for feature type', ftype, 'in R!'))
			return(FALSE)
		}

		sg('set_features', 'TRAIN', data_train)
		sg('set_features', 'TEST', data_test)
	}

	else {
		sg('set_features', 'TRAIN', kernel_data)
		sg('set_features', 'TEST', kernel_data)
	}

	convert_features_and_add_preproc(prefix)

	return(TRUE)
}
