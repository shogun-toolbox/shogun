from tools.load import LoadMatrix

lm=LoadMatrix()
data = lm.load_numbers('../data/fm_train_real.dat')

parameter_list = [[data]]

def preprocessor_classicisomap_modular(data):
	from shogun.Features import RealFeatures
	from shogun.Preprocessor import ClassicIsomap
	
	features = RealFeatures(data)
		
	preprocessor = ClassicIsomap()
	preprocessor.set_target_dim(1)
	preprocessor.apply_to_feature_matrix(features)

	return features


if __name__=='__main__':
	print 'ClassicIsomap'
	preprocessor_classicisomap_modular(*parameter_list[0])

