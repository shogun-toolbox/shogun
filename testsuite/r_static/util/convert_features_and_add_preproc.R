convert_features_and_add_preproc <- function(prefix) {
	if (!exists(paste(prefix, 'order', sep=''), mode='numeric')) {
		return(FALSE)
	}

	ftype <- eval(parse(text=paste(prefix, 'feature_type', sep='')))
	if (regexpr('Ulong', ftype)>0) {
		type <- 'ULONG'
	} else if (regexpr('Word', ftype)>0) {
		type <- 'WORD'
	} else {
		return(FALSE)
	}

	order <- eval(parse(text=paste(prefix, 'order', sep='')))
	gap <- eval(parse(text=paste(prefix, 'gap', sep='')))
	reverse <- eval(parse(text=paste(prefix, 'reverse', sep='')))
	sg('add_preproc', paste('SORT', type, 'STRING', sep=''))
	sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', type,
		order, order-1, gap, reverse)
	sg('attach_preproc', 'TRAIN')

	sg('convert', 'TEST', 'STRING', 'CHAR', 'STRING', type,
		order, order-1, gap, reverse)
	sg('attach_preproc', 'TEST')

	return(TRUE)
}
