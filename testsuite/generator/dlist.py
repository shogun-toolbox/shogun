# format:
#	name:
#		data_class + _type, feature_class + _type(s),
#		available kernel parameters, accuracy
DLIST={
	'CanberraMetric':[['rand', 'double'], ['simple', 'Real'], [], 1e-8],
	'ChebyshewMetric':[['rand', 'double'], ['simple', 'Real'], [], 1e-8],
	'CanberraWordDistance':[['dna', ''], ['string_complex', 'Word', 'Char'], [], 1e-7],
	'EuclidianDistance':[['rand', 'double'], ['simple', 'Real'], [], 1e-8],
	'GeodesicMetric':[['rand', 'double'], ['simple', 'Real'], [], 1e-8],
	'HammingWordDistance':[['dna', ''], ['string_complex', 'Word', 'Char'], ['use_sign'], 0],
	'JensenMetric':[['rand', 'double'], ['simple', 'Real'], [], 1e-8],
	'ManhattanMetric':[['rand', 'double'], ['simple', 'Real'], [], 1e-8],
	'ManhattanWordDistance':[['dna', ''], ['string_complex', 'Word', 'Char'], [], 0],
	'MinkowskiMetric':[['rand', 'double'], ['simple', 'Real'], ['k'], 1e-8],
	'SparseEuclidianDistance':[['rand', 'double'], ['simple', 'Real'], [], 1e-7],
}
