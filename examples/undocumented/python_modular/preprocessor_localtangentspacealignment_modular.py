from tools.load import LoadMatrix

lm=LoadMatrix()
data = lm.load_numbers('../data/fm_train_real.dat')

parameter_list = [[data,10],[data,20]]

def preprocessor_localtangentspacealignment_modular(data,k):
	from shogun.Features import RealFeatures
	from shogun.Preprocessor import LocalTangentSpaceAlignment
	
	features = RealFeatures(data)
		
	preprocessor = LocalTangentSpaceAlignment()
	preprocessor.set_target_dim(1)
	preprocessor.set_k(k)
	preprocessor.apply_to_feature_matrix(features)

	return features


if __name__=='__main__':
	print 'LocalTangentSpaceAlignment'
	preprocessor_localtangentspacealignment_modular(*parameter_list[0])

