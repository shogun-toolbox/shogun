set_features <- function() {
	source('util/convert_features_and_add_preproc.R')

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

	if (exists('name_features')) {
		print(paste('Features', name_features, 'not supported yet!'))
		return(FALSE)
	}

	sg('clean_features', 'TRAIN')
	sg('clean_features', 'TEST')

	if (regexpr('Combined', name)>0) {
		if (exists('subkernel0_alphabet')) {
			sg('add_features', 'TRAIN',
				subkernel0_data_train, subkernel0_alphabet)
			sg('add_features', 'TEST',
				subkernel0_data_test, subkernel0_alphabet)
		} else {
			sg('add_features', 'TRAIN', subkernel0_data_train)
			sg('add_features', 'TEST', subkernel0_data_test)
		}

		if (exists('subkernel1_alphabet')) {
			sg('add_features', 'TRAIN',
				subkernel1_data_train, subkernel1_alphabet)
			sg('add_features', 'TEST',
				subkernel1_data_test, subkernel1_alphabet)
		} else {
			sg('add_features', 'TRAIN', subkernel1_data_train)
			sg('add_features', 'TEST', subkernel1_data_test)
		}

		if (exists('subkernel2_alphabet')) {
			sg('add_features', 'TRAIN',
				subkernel2_data_train, subkernel2_alphabet)
			sg('add_features', 'TEST',
				subkernel2_data_test, subkernel2_alphabet)
		} else {
			sg('add_features', 'TRAIN', subkernel2_data_train)
			sg('add_features', 'TEST', subkernel2_data_test)
		}
	}

	else if (exists('alphabet')) {
		sg('set_features', 'TRAIN', data_train, alphabet)
		sg('set_features', 'TEST', data_test, alphabet)
	}

	else if (exists('data_train')) {
		if (regexpr('double', data_type)<0) {
			print(paste('No support for data type', data_type, 'in R!'))
			return(FALSE)
		}

		sg('set_features', 'TRAIN', data_train)
		sg('set_features', 'TEST', data_test)
	}

	else {
		sg('set_features', 'TRAIN', data)
		sg('set_features', 'TEST', data)
	}

	convert_features_and_add_preproc()

	return(TRUE)
}
