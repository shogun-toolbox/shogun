fix_distance_name_inconsistency <- function(dname) {
	dname=toupper(dname)
	if (regexpr('WORDDISTANCE', dname)>0) {
		return(strsplit(dname, 'WORDDISTANCE')[[1]])
	} else if (regexpr('DISTANCE', dname)>0) {
		return(strsplit(dname, 'DISTANCE')[[1]])
	} else if (regexpr('METRIC', dname)>0) {
		return(strsplit(dname, 'METRIC')[[1]])
	} else {
		return(dname);
	}
}
