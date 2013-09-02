#!/usr/bin/env python
from tools.load import LoadMatrix

lm=LoadMatrix()
data = lm.load_numbers('../data/fm_train_real.dat')

parameter_list = [[data]]

def preprocessor_pca_modular (data):
	from modshogun import RealFeatures
	from modshogun import PCA
	
	features = RealFeatures(data)
		
	preprocessor = PCA()
	preprocessor.init(features)
	preprocessor.apply_to_feature_matrix(features)

	return features


if __name__=='__main__':
	print('PCA')
	preprocessor_pca_modular(*parameter_list[0])

