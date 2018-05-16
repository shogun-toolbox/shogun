#!/usr/bin/env python
from tools.load import LoadMatrix

lm=LoadMatrix()
data = lm.load_numbers('../data/fm_train_real.dat')

parameter_list = [[data]]

def preprocessor_pca (data):
	from shogun import RealFeatures
	from shogun import PCA

	features = RealFeatures(data)

	preprocessor = PCA()
	preprocessor.fit(features)
	features = preprocessor.apply(features)

	return features


if __name__=='__main__':
	print('PCA')
	preprocessor_pca(*parameter_list[0])

