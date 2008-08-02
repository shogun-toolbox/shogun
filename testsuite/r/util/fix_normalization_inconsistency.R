fix_normalization_inconsistency <- function(normalization) {
	if (normalization==1) {
		return('SQRT')
	} else if (normalization==2) {
		return('FULL')
	} else if (normalization==3) {
		return('SQRTLEN')
	} else if (normalization==4) {
		return('LEN')
	} else if (normalization==5) {
		return('SQLEN')
	} else {
		return('NO')
	}
}
