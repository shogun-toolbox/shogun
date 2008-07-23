function y = set_distance()
	global distance_name;
	global name;
	global feats_train;
	global feats_test;
	global distance;

	if !isempty(distance_name)
		dname=distance_name;
	else
		dname=name;
	end

	if strcmp(dname, 'CanberraMetric')==1
		distance=CanberraMetric(feats_train, feats_train);

	elseif strcmp(dname, 'CanberraWordDistance')==1
		distance=CanberraWordDistance(feats_train, feats_train);

	elseif strcmp(dname, 'ChebyshewMetric')==1
		distance=ChebyshewMetric(feats_train, feats_train);

	elseif strcmp(dname, 'EuclidianDistance')==1
		distance=EuclidianDistance(feats_train, feats_train);

	elseif strcmp(dname, 'GeodesicMetric')==1
		distance=GeodesicMetric(feats_train, feats_train);

	elseif strcmp(dname, 'HammingWordDistance')==1
		global distance_arg0_use_sign;
		distance=HammingWordDistance(feats_train, feats_train, tobool(distance_arg0_use_sign));

	elseif strcmp(dname, 'JensenMetric')==1
		distance=JensenMetric(feats_train, feats_train);

	elseif strcmp(dname, 'ManhattanMetric')==1
		distance=ManhattanMetric(feats_train, feats_train);

	elseif strcmp(dname, 'ManhattanWordDistance')==1
		distance=ManhattanWordDistance(feats_train, feats_train);

	elseif strcmp(dname, 'MinkowskiMetric')==1
		global distance_arg0_k;
		distance=MinkowskiMetric(feats_train, feats_train, distance_arg0_k);

	elseif strcmp(dname, 'SparseEuclidianDistance')==1
		distance=SparseEuclidianDistance(feats_train, feats_train);

	else
		printf("Unknown distance %s!\n", dname);
		y=false;
		return;
	end

	y=true;
