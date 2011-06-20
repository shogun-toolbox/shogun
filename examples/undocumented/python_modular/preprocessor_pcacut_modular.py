from tools.load import LoadMatrix

lm=LoadMatrix()
data = lm.load_numbers('../data/fm_train_real.dat')

parameter_list = [[data]]

def preprocessor_pcacut_modular(data):
	from shogun.Features import RealFeatures
	from shogun.Preprocessor import PCACut
	
	features = RealFeatures(data)
		
	preprocessor = PCACut()
	preprocessor.init(features)
	preprocessor.apply_to_feature_matrix(features)

	return features


if __name__=='__main__':
	print 'PCACut'
	preprocessor_pcacut_modular(*parameter_list[0])

