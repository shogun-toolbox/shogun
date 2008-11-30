function y = set_distance()
	global distance_name;
	global feats_train;
	global feats_test;
	global distance;
	global kernel_arg2_distance;
	y=false;

	if ~isempty(kernel_arg2_distance)
		dname=kernel_arg2_distance;
	else
		dname=distance_name;
	end

	if strcmp(dname, 'BrayCurtisDistance')==1
		global BrayCurtisDistance;
		distance=BrayCurtisDistance(feats_train, feats_train);
	elseif strcmp(dname, 'CanberraMetric')==1
		global CanberraMetric;
		distance=CanberraMetric(feats_train, feats_train);

	elseif strcmp(dname, 'CanberraWordDistance')==1
		global CanberraWordDistance;
		distance=CanberraWordDistance(feats_train, feats_train);

	elseif strcmp(dname, 'ChebyshewMetric')==1
		global ChebyshewMetric;
		distance=ChebyshewMetric(feats_train, feats_train);

	elseif strcmp(dname, 'ChiSquareDistance')==1
		global ChiSquareDistance;
		distance=ChiSquareDistance(feats_train, feats_train);

	elseif strcmp(dname, 'CosineDistance')==1
		global CosineDistance;
		distance=CosineDistance(feats_train, feats_train);

	elseif strcmp(dname, 'EuclidianDistance')==1
		global EuclidianDistance;
		distance=EuclidianDistance(feats_train, feats_train);

	elseif strcmp(dname, 'GeodesicMetric')==1
		global GeodesicMetric;
		distance=GeodesicMetric(feats_train, feats_train);

	elseif strcmp(dname, 'HammingWordDistance')==1
		global HammingWordDistance;
		global distance_arg0_use_sign;
		distance=HammingWordDistance(feats_train, feats_train, ...
			tobool(distance_arg0_use_sign));

	elseif strcmp(dname, 'JensenMetric')==1
		global JensenMetric;
		distance=JensenMetric(feats_train, feats_train);

	elseif strcmp(dname, 'ManhattanMetric')==1
		global ManhattanMetric;
		distance=ManhattanMetric(feats_train, feats_train);

	elseif strcmp(dname, 'ManhattanWordDistance')==1
		global ManhattanWordDistance;
		distance=ManhattanWordDistance(feats_train, feats_train);

	elseif strcmp(dname, 'MinkowskiMetric')==1
		global MinkowskiMetric;
		global distance_arg0_k;
		distance=MinkowskiMetric(feats_train, feats_train, distance_arg0_k);

	elseif strcmp(dname, 'SparseEuclidianDistance')==1
		global SparseEuclidianDistance;
		distance=SparseEuclidianDistance(feats_train, feats_train);

	elseif strcmp(dname, 'TanimotoDistance')==1
		global TanimotoDistance;
		distance=TanimotoDistance(feats_train, feats_train);

	else
		error('Unknown distance %s!', dname);
	end

	y=true;
