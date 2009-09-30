set_distance <- function() {
	source('util/tobool.R')
	source('util/fix_distance_name_inconsistency.R')

	dname <- fix_distance_name_inconsistency(distance_name)
	ftype <- toupper(distance_feature_type)

	if (regexpr('HAMMING', dname)>0) {
		sg('set_distance', dname, ftype, tobool(distance_arg0_use_sign));
	} else if (regexpr('MINKOWSKI', dname)>0) {
		sg('set_distance', dname, ftype, distance_arg0_k)
	} else {
		sg('set_distance', dname, ftype)
	}

	return(TRUE)
}
