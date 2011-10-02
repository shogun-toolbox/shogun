from tools.load import LoadMatrix

lm=LoadMatrix()
data = lm.load_numbers('../data/fm_train_real.dat')

parameter_list = [[data,10],[data,20]]

def preprocessor_kernellocaltangentspacealignment_modular(data,k):
	from shogun.Features import RealFeatures
	from shogun.Preprocessor import KernelLocalTangentSpaceAlignment
	
	features = RealFeatures(data)
		
	preprocessor = KernelLocalTangentSpaceAlignment()
	preprocessor.set_target_dim(1)
	preprocessor.set_k(k)
	preprocessor.apply_to_feature_matrix(features)

	return features


if __name__=='__main__':
	print 'KernelLocalTangentSpaceAlignment'
	preprocessor_kernellocaltangentspacealignment_modular(*parameter_list[0])

