function y = set_distance()
	global distance_name;
	global name;
	global feats_train;
	global feats_test;
	global dist;
	y=false;

	if isempty(distance_name)
		dname=name;
	else
		dname=distance_name;
	end

	if strcmp(dname, 'CanberraMetric')==1
		global CanberraMetric;
		dist=CanberraMetric(feats_train, feats_train);

	elseif strcmp(dname, 'CanberraWordDistance')==1
		global CanberraWordDistance;
		dist=CanberraWordDistance(feats_train, feats_train);

	elseif strcmp(dname, 'ChebyshewMetric')==1
		global ChebyshewMetric;
		dist=ChebyshewMetric(feats_train, feats_train);

	elseif strcmp(dname, 'EuclidianDistance')==1
		global EuclidianDistance;
		dist=EuclidianDistance(feats_train, feats_train);

	elseif strcmp(dname, 'GeodesicMetric')==1
		global GeodesicMetric;
		dist=GeodesicMetric(feats_train, feats_train);

	elseif strcmp(dname, 'HammingWordDistance')==1
		global HammingWordDistance;
		global distance_arg0_use_sign;
		dist=HammingWordDistance(feats_train, feats_train, tobool(distance_arg0_use_sign));

	elseif strcmp(dname, 'JensenMetric')==1
		global JensenMetric;
		dist=JensenMetric(feats_train, feats_train);

	elseif strcmp(dname, 'ManhattanMetric')==1
		global ManhattanMetric;
		dist=ManhattanMetric(feats_train, feats_train);

	elseif strcmp(dname, 'ManhattanWordDistance')==1
		global ManhattanWordDistance;
		dist=ManhattanWordDistance(feats_train, feats_train);

	elseif strcmp(dname, 'MinkowskiMetric')==1
		global MinkowskiMetric;
		global distance_arg0_k;
		dist=MinkowskiMetric(feats_train, feats_train, distance_arg0_k);

	elseif strcmp(dname, 'SparseEuclidianDistance')==1
		global SparseEuclidianDistance;
		dist=SparseEuclidianDistance(feats_train, feats_train);

	else
		error('Unknown distance %s!', dname);
	end

	y=true;
