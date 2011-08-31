fix_classifier_name_inconsistency <- function(cname) {
	cname <- toupper(cname)

	if (regexpr('LIBSVM', cname)>0) {
		if (nchar(cname)>nchar('LIBSVM')) {
			ending <- strsplit(cname, 'LIBSVM')[[1]][2]
			return(paste('LIBSVM_', ending, sep=''))
		}
	}

	return(cname)
}
