get_distance <- function(feats) {
	source('util/tobool.R')

	if (exists('kernel_arg2_distance')) {
		dname <- kernel_arg2_distance
	} else {
		dname <- distance_name
	}

	if (regexpr('HammingWordDistance', dname)>0) {
		distance <- HammingWordDistance(feats[[1]], feats[[1]],
			tobool(distance_arg0_use_sign))
	} else if (regexpr('MinkowskiMetric', dname)>0) {
		distance <- MinkowskiMetric(feats[[1]], feats[[1]], distance_arg0_k)
	} else {
		dfun <- eval(parse(text=dname))
		distance <- dfun(feats[[1]], feats[[1]])
	}

	return(distance)
}
