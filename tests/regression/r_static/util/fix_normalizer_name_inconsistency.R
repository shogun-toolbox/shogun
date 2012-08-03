fix_normalizer_name_inconsistency <- function(name) {
	if (regexpr('Identity', name)>0) {
		return('IDENTITY')
	} else if (regexpr('AvgDiag', name)>0) {
		return('AVGDIAG')
	} else if (regexpr('SqrtDiag', name)>0) {
		return('SQRTDIAG')
	} else if (regexpr('FirstElement', name)>0) {
		return('FIRSTELEMENT')
	} else {
		return(FALSE)
	}
}
