set_distance <- function() {
	source('util/tobool.R')
	source('util/fix_distance_name_inconsistency.R')

	if (exists('distance_name')) {
		dname <- fix_distance_name_inconsistency(distance_name)
	} else {
		dname <- fix_distance_name_inconsistency(name)
	}

	ftype <- toupper(feature_type)

	if (regexpr('HAMMING', dname)>0) {
		sg('set_distance', dname, ftype, tobool(distance_arg0_use_sign));
	} else if (regexpr('MINKOWSKI', dname)>0) {
		sg('set_distance', dname, ftype, distance_arg0_k)
	} else {
		sg('set_distance', dname, ftype)
	}

	sg('init_distance', 'TRAIN')
	return(TRUE)
}
