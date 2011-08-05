from tools.load import LoadMatrix

lm=LoadMatrix()
data = lm.load_numbers('../data/fm_train_real.dat')

parameter_list = [[data,10],[data,20]]

def preprocessor_laplacianeigenmaps_modular(data,k):
	from shogun.Features import RealFeatures
	from shogun.Preprocessor import LaplacianEigenmaps
	
	features = RealFeatures(data)
		
	preprocessor = LaplacianEigenmaps()
	preprocessor.set_target_dim(1)
	preprocessor.set_k(k)
	preprocessor.set_tau(2.0)
	preprocessor.apply_to_feature_matrix(features)

	return features


if __name__=='__main__':
	print 'LaplacianEigenmaps'
	preprocessor_laplacianeigenmaps_modular(*parameter_list[0])

