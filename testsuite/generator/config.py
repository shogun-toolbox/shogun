"""
Global configuration parameters: lists and types
"""

# name: accuracy, classifier type, label type
CLASSIFIER={
	'SVMLight':[1e-6, 'kernel', 'twoclass'],
	'LDA':[1e-7, 'lda', 'twoclass'],
	'LibLinear':[1e-8, 'linear', 'twoclass'],
	'LibSVM':[1e-3, 'kernel', 'twoclass'],
	'LibSVMMultiClass':[1e-8, 'kernel', 'series'],
	'LibSVMOneClass':[1e-5, 'kernel', None],
	'GMNPSVM':[1e-8, 'kernel', 'series'],
	'GPBTSVM':[1e-6, 'kernel', 'twoclass'],
	'KNN':[1e-8, 'knn', 'twoclass'],
	'MPDSVM':[1e-6, 'kernel', 'twoclass'],
	'Perceptron':[1e-8, 'perceptron', 'twoclass'],
	'SubGradientSVM':[1e-6, 'linear', 'twoclass'],
	'SVMLin':[1e-7, 'linear', 'twoclass'],
	'SVMOcas':[1e-4, 'linear', 'twoclass'],
}

# name: accuracy
CLUSTERING={
	'Hierarchical':[1e-8],
	'KMeans':[1e-8],
}

# name: data_class + _type, feature_class + _type(s), available distance
#  parameters, accuracy
DISTANCE={
	'CanberraMetric':[['rand', 'double'], ['simple', 'Real'], [], 1e-8],
	'ChebyshewMetric':[['rand', 'double'], ['simple', 'Real'], [], 1e-8],
	'CanberraWordDistance':[['dna', ''], ['string_complex', 'Word', 'Char'],
		[], 1e-7],
	'EuclidianDistance':[['rand', 'double'], ['simple', 'Real'], [], 1e-8],
	'GeodesicMetric':[['rand', 'double'], ['simple', 'Real'], [], 1e-8],
	'HammingWordDistance':[['dna', ''], ['string_complex', 'Word', 'Char'],
		['use_sign'], 0],
	'JensenMetric':[['rand', 'double'], ['simple', 'Real'], [], 1e-8],
	'ManhattanMetric':[['rand', 'double'], ['simple', 'Real'], [], 1e-8],
	'ManhattanWordDistance':[['dna', ''], ['string_complex', 'Word', 'Char'],
		[], 0],
	'MinkowskiMetric':[['rand', 'double'], ['simple', 'Real'], ['k'], 1e-8],
	'SparseEuclidianDistance':[['rand', 'double'], ['simple', 'Real'],
		[], 1e-7],
}

# name: data_class + _type, feature_class + _type(s), accuracy
DISTRIBUTION={
	'Histogram':[['rand', 'ushort'], ['simple', 'Word'], 1e-8],
	'HMM':[['dna', ''], ['string', 'Char'], 1e-9],
}

# name: data_class + _type, feature_class + _type(s), available kernel
#  parameters, accuracy
KERNEL={
	'AUC':[['rand', 'ushort'], ['simple', 'Word'], ['subkernel'], 1e-8],
	'Byte':[['rand', 'ubyte'], ['simple', 'Byte'], [], 1e-8],
	'Char':[['rand', 'character'], ['simple', 'Char'], [], 1e-8],
	'Chi2':[['rand', 'double'], ['simple', 'Real'], ['width', 'size'], 1e-8],
	'Combined':[['', ''], ['', ''], ['append_subkernel_weights'], 1e-8],
	'CommUlongString':[['dna', ''], ['string_complex', 'Ulong', 'Char'],
		['use_sign', 'normalization'], 1e-9],
	'CommWordString':[['dna', ''], ['string_complex', 'Word', 'Char'],
		['use_sign', 'normalization'], 1e-9],
	'Const':[['rand', 'double'], ['simple', 'Real'], ['c'], 0],
	'Custom':[['rand', 'double'], ['custom', ''], [], 1e-6],
	'Diag':[['rand', 'double'], ['simple', 'Real'], ['diag'], 0],
	'Distance':[['rand', 'double'], ['simple', 'Real'],
		['width', 'distance'], 1e-9],
	'FixedDegreeString':[['dna', ''], ['string', 'Char'], ['degree'], 1e-9],
	'Gaussian':[['rand', 'double'], ['simple', 'Real'], ['width'], 1e-6],
	'GaussianShift':[['rand', 'double'], ['simple', 'Real'],
		['width', 'max_shift', 'shift_step'], 1e-8],
	'HistogramWord':[['rand', 'ushort'], ['simple', 'Word'],
		[], 1e-6],
	'Int':[['rand', 'int'], ['simple', 'Int'], [], 1e-8],
	'Linear':[['rand', 'double'], ['simple', 'Real'], ['scale'], 1e-8],
	'LinearByte':[['rand', 'ubyte'], ['simple', 'Byte'],
		['do_rescale', 'scale'], 1e-8],
	'LinearString':[['dna', ''], ['string', 'Char'], ['scale'], 1e-8],
	'LinearWord':[['rand', 'ushort'], ['simple', 'Word'],
		['do_rescale', 'scale'], 1e-8],
	'LocalAlignmentString':[['dna', ''], ['string', 'Char'], [], 1e-8],
	'LocalityImprovedString':[['dna', ''], ['string', 'Char'],
		['length', 'inner_degree', 'outer_degree'], 1e-8],
	'MindyGram':[['', ''], ['mindy', ''], ['measure', 'width'], 1e-8],
	'Poly':[['rand', 'double'], ['simple', 'Real'],
		['degree', 'inhomogene', 'use_normalization'], 1e-6],
	'PolyMatchString':[['dna', ''], ['string', 'Char'],
		['degree', 'inhomogene'], 1e-10],
	'PolyMatchWord':[['rand', 'ushort'], ['simple', 'Word'],
		['degree', 'inhomogene'], 1e-10],
	'Real':[['rand', 'double'], ['simple', 'Real'], [], 1e-8],
	'SalzbergWord':[['rand', 'ushort'], ['simple', 'Word'],
		[], 1e-6],
	'Short':[['rand', 'ushort'], ['simple', 'short'], [], 1e-8],
	'Sigmoid':[['rand', 'double'], ['simple', 'Real'],
		['size', 'gamma_', 'coef0'], 1e-9],
	'SimpleLocalityImprovedString':[['dna', ''], ['string', 'Char'],
		['length', 'inner_degree', 'outer_degree'], 1e-15],
	'SparseReal':[['rand', 'double'], ['simple', 'Real'], [], 1e-8],
	'SparseGaussian':[['rand', 'double'], ['simple', 'Real'],
		['width'], 1e-8],
	'SparseLinear':[['rand', 'double'], ['simple', 'Real'],
		['scale'], 1e-8],
	'SparsePoly':[['rand', 'double'], ['simple', 'Real'],
		['size', 'degree', 'inhomogene', 'use_normalization'], 1e-8],
	'SparseWord':[['rand', 'double'], ['simple', 'Word'], [], 1e-8],
	'StringCharKernel':[['dna', ''], ['string', 'Char'], [], 1e-8],
	'StringIntKernel':[['dna', ''], ['string', 'Int'], [], 1e-8],
	'StringRealKernel':[['dna', ''], ['string', 'Real'], [], 1e-8],
	'StringUlongKernel':[['dna', ''], ['string', 'Ulong'], [], 1e-8],
	'StringWordKernel':[['dna', ''], ['string', 'Word'], [], 1e-8],
	'Ulong':[['rand', 'double'], ['simple', 'Ulong'], [], 1e-8],
	'WeightedDegreePositionString':[['dna', ''], ['string', 'Char'],
		['degree', 'weights', 'shift', 'shift_len', 'max_mismatch',
		'use_normalization', 'mkl_stepsize'], 1e-8],
	'WeightedDegreeString':[['dna', ''], ['string', 'Char'], ['degree'], 1e-10],
	'WeightedCommWordString':[['dna', ''], ['string_complex', 'Word', 'Char'],
		['use_sign', 'normalization'], 1e-9],
	'Word':[['rand', 'ushort'], ['simple', 'Word'], [], 1e-8],
	'WordMatch':[['rand', 'ushort'], ['simple', 'Word'],
		['degree', 'do_rescale', 'scale'], 1e-8],
}

# name: accuracy, classifier type
REGRESSION={
	'SVRLight':[1e-6, 'svm'],
	'LibSVR':[1e-6, 'svm'],
	'KRR':[1e-8, 'kernelmachine'],
}

C_KERNEL=0
C_DISTANCE=1
C_CLASSIFIER=2
C_CLUSTERING=3
C_DISTRIBUTION=4
C_REGRESSION=5
