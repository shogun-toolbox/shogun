function y = set_and_train_distance()
	global distance_name;
	global name;
	global feature_type;

	dargs=0;

	if !isempty(distance_name)
		dname=fix_distance_name_inconsistency(distance_name);
	else
		dname=fix_distance_name_inconsistency(name);
	end

	sg('set_distance', dname, toupper(feature_type), dargs);
	sg('init_distance', 'TRAIN');
	y=true;
