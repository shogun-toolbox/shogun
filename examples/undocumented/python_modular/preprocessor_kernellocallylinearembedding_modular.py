from tools.load import LoadMatrix

lm=LoadMatrix()
data = lm.load_numbers('../data/fm_train_real.dat')

parameter_list = [[data,10],[data,20]]

def preprocessor_kernellocallylinearembedding_modular(data,k):
	from shogun.Features import RealFeatures
	from shogun.Preprocessor import KernelLocallyLinearEmbedding
	from shogun.Kernel import LinearKernel
	
	features = RealFeatures(data)
		
	kernel = LinearKernel()

	preprocessor = KernelLocallyLinearEmbedding(kernel)
	preprocessor.set_target_dim(1)
	preprocessor.set_k(k)
	preprocessor.apply_to_feature_matrix(features)

	return features


if __name__=='__main__':
	print 'KernelLocallyLinearEmbedding'
	preprocessor_kernellocallylinearembedding_modular(*parameter_list[0])

