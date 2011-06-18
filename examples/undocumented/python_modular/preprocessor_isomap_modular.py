from numpy import random

random.seed(17)
data = random.randn(10,100)

parameter_list = [[data]]

def preprocessor_isomap_modular(data):
	from shogun.Features import RealFeatures
	from shogun.Preprocessor import Isomap
	
	features = RealFeatures(data)
		
	preprocessor = Isomap()
	preprocessor.set_target_dim(3)
	preprocessor.apply_to_feature_matrix(features)

	return features


if __name__=='__main__':
	print 'Isomap'
	preprocessor_isomap_modular(*parameter_list[0])

