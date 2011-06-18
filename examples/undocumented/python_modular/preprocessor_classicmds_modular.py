from numpy import random

random.seed(17)
data = random.randn(10,100)

parameter_list = [[data]]

def preprocessor_classicmds_modular(data):
	from shogun.Features import RealFeatures
	from shogun.Preprocessor import ClassicMDS
	
	features = RealFeatures(data)
		
	preprocessor = ClassicMDS()
	preprocessor.set_target_dim(3)
	preprocessor.apply_to_feature_matrix(features)

	return features


if __name__=='__main__':
	print 'ClassicMDS'
	preprocessor_classicmds_modular(*parameter_list[0])

