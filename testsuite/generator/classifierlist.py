# format:
#	name: accuracy, classifier type (see fileop.py), label type
CLASSIFIERLIST={
	'SVMLight':[1e-7, 'kernel', 'twoclass'],
	'LibLinear':[1e-8, 'linear', 'twoclass'],
	'LibSVM':[1e-8, 'kernel', 'twoclass'],
	'LibSVMMultiClass':[1e-8, 'kernel', 'series'],
	'LibSVMOneClass':[1e-8, 'kernel', None],
	'GMNPSVM':[1e-8, 'kernel', 'series'],
	'GPBTSVM':[1e-7, 'kernel', 'twoclass'],
	'KNN':[1e-8, 'distance', 'twoclass'],
	'MPDSVM':[1e-8, 'kernel', 'twoclass'],
	'Perceptron':[1e-8, 'perceptron', 'twoclass'],
	'SubGradientSVM':[1e-8, 'linear', 'twoclass'],
	'SVMLin':[1e-8, 'linear', 'twoclass'],
	'SVMOcas':[1e-4, 'linear', 'twoclass'],
}
