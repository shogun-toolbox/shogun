function y = fix_classifier_name_inconsistency (cname)
	cname=toupper(cname)
	if findstr('LIBSVM', cname) && length(cname)>length('LIBSVM')
		pos=findstr('LIBSVM', cname)
		y=strcat('LIBSVM_', cname(pos+6:end))
	else
		y=cname
	end
