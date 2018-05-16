#!/usr/bin/env python
from tools.load import LoadMatrix

lm=LoadMatrix()
data = lm.load_numbers('../data/fm_train_real.dat')

parameter_list = [[data, 20], [data, 30]]

def preprocessor_dimensionreductionpreprocessor (data, k):
	from shogun import RealFeatures
	from shogun import DimensionReductionPreprocessor
	try:
		from shogun import LocallyLinearEmbedding
	except ImportError:
		print("LocallyLinearEmbedding not available")
		exit(0)

	features = RealFeatures(data)

	converter = LocallyLinearEmbedding()
	converter.set_k(k)

	preprocessor = DimensionReductionPreprocessor(converter)
	preprocessor.fit(features)
	features = preprocessor.apply(features)

	return features


if __name__=='__main__':
	print('DimensionReductionPreprocessor')
	preprocessor_dimensionreductionpreprocessor(*parameter_list[0])

