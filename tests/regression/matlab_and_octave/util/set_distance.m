function y = set_distance()
	global distance_name;
	global distance_feature_type;
	ftype=upper(distance_feature_type);
	dname=fix_distance_name_inconsistency(distance_name);

	if strcmp(dname, 'HAMMING')==1
		global distance_arg0_use_sign;
		sg('set_distance', dname, ftype, tobool(distance_arg0_use_sign));
	elseif strcmp(dname, 'MINKOWSKI')==1
		global distance_arg0_k;
		sg('set_distance', dname, ftype, distance_arg0_k);
	elseif strcmp(dname, 'SPARSEEUCLIDEAN')==1
		sg('set_distance', 'EUCLIDEAN', 'SPARSEREAL');
	else
		sg('set_distance', dname, ftype);
	end

	y=true;
