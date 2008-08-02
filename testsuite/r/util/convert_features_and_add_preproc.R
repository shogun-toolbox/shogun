convert_features_and_add_preproc <- function() {
	if (!exists('order', mode='numeric')) {
		return(FALSE)
	}

	if (regexpr('Ulong', feature_type)>0) {
		type <- 'ULONG'
	} else if (regexpr('Word', feature_type)>0) {
		type='WORD'
	} else {
		return(FALSE)
	}

	sg('add_preproc', paste('SORT', type, 'STRING', sep=''))
	sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', type,
		order, order-1, gap, reverse)
	sg('attach_preproc', 'TRAIN')

	sg('convert', 'TEST', 'STRING', 'CHAR', 'STRING', type,
		order, order-1, gap, reverse)
	sg('attach_preproc', 'TEST')

	return(TRUE)
}
