from numpy import random
from tools.load import LoadMatrix

lm=LoadMatrix()
data = lm.load_numbers('../data/fm_train_real.dat')

parameter_list = [[data]]

def preprocessor_classicmds_modular(data):
	from shogun.Features import RealFeatures
	from shogun.Preprocessor import ClassicMDS
	
	features = RealFeatures(data)
		
	preprocessor = ClassicMDS()
	preprocessor.set_target_dim(1)
	preprocessor.apply_to_feature_matrix(features)

	return features


if __name__=='__main__':
	print 'ClassicMDS'
	preprocessor_classicmds_modular(*parameter_list[0])

