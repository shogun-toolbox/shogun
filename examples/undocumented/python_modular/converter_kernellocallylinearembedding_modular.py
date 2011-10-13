from tools.load import LoadMatrix

lm=LoadMatrix()
data = lm.load_numbers('../data/fm_train_real.dat')

parameter_list = [[data,10],[data,20]]

def converter_kernellocallylinearembedding_modular(data,k):
	from shogun.Features import RealFeatures
	from shogun.Converter import KernelLocallyLinearEmbedding
	from shogun.Kernel import LinearKernel
	
	features = RealFeatures(data)
		
	kernel = LinearKernel()

	converter = KernelLocallyLinearEmbedding(kernel)
	converter.set_target_dim(1)
	converter.set_k(k)
	converter.apply(features)

	return features


if __name__=='__main__':
	print 'KernelLocallyLinearEmbedding'
	converter_kernellocallylinearembedding_modular(*parameter_list[0])

