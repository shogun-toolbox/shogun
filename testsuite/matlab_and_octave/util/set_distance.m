function y = set_distance()
	global distance_name;
	global name;
	global feature_type;
	ftype=upper(feature_type);

	if ~isempty(distance_name)
		dname=fix_distance_name_inconsistency(distance_name);
	else
		dname=fix_distance_name_inconsistency(name);
	end

	if strcmp(dname, 'HAMMING')==1
		global distance_arg0_use_sign;
		sg('set_distance', dname, ftype, tobool(distance_arg0_use_sign));
	elseif strcmp(dname, 'MINKOWSKI')==1
		global distance_arg0_k;
		sg('set_distance', dname, ftype, distance_arg0_k);
	else
		sg('set_distance', dname, ftype);
	end

	sg('init_distance', 'TRAIN');
	y=true;
