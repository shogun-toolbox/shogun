fix_kernel_name_inconsistency = function(kname) {
	kname=toupper(kname);

	if (regexpr('SIMPLELOCALITYIMPROVEDSTRING', kname)>0) {
		return('SLIK')
	}
	else if (regexpr('LOCALITYIMPROVEDSTRING', kname)>0) {
		return('LIK')
	}
	else if (regexpr('WORDMATCH', kname)>0) {
		return('MATCH')
	}
	else if (regexpr('WEIGHTEDDEGREEPOSITIONSTRING', kname)>0) {
		return('WEIGHTEDDEGREEPOS')
	}
	else if (regexpr('WEIGHTEDCOMMWORDSTRING', kname)>0) {
		return('WEIGHTEDCOMMSTRING')
	}
	else if (regexpr('COMMULONGSTRING', kname)>0) {
		return('COMMSTRING')
	}
	else if (regexpr('COMMWORDSTRING', kname)>0) {
		return('COMMSTRING')
	}
	else if (regexpr('WORDSTRING', kname)>0) {
		return(strsplit(kname, 'WORDSTRING')[[1]])
	}
	else if (regexpr('STRING', kname)>0) {
		return(strsplit(kname, 'STRING')[[1]])
	}
	else if (regexpr('WORD', kname)>0) {
		return(strsplit(kname, 'WORD')[[1]])
	}
	else {
		return(kname);
	}

}
