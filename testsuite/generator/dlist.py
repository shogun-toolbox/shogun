# format:
#	name:
#		data_class + _type, feature_class + _type(s),
#		available kernel parameters, accuracy
DLIST={
	'CanberraWordDistance':[['dna', ''], ['string_complex', 'Word', 'Char'], [], 1e-7],
	'EuclidianDistance':[['rand', 'double'], ['simple', 'Real'], [], 1e-8],
	'HammingWordDistance':[['dna', ''], ['string_complex', 'Word', 'Char'], ['use_sign'], 0],
	'ManhattanWordDistance':[['dna', ''], ['string_complex', 'Word', 'Char'], [], 0],
	'SparseEuclidianDistance':[['rand', 'double'], ['simple', 'Real'], [], 1e-7],
}
