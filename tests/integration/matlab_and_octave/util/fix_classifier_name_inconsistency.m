function y = fix_classifier_name_inconsistency (cname)
	cname=upper(cname);

	if findstr('LIBSVM', cname)
		if length(cname) > length('LIBSVM')
			pos=findstr('LIBSVM', cname);
			y=strcat('LIBSVM_', cname(pos+6:end));
			return;
		end
	elseif findstr('LIBLINEAR', cname)
		y='LIBLINEAR_L2R_LR';
		return
	end

	y=cname;
