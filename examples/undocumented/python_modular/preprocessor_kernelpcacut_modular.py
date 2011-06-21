from tools.load import LoadMatrix

lm=LoadMatrix()
data = lm.load_numbers('../data/fm_train_real.dat')

parameter_list = [[data, 0.01, 1.0], [data, 0.05, 2.0]]

def preprocessor_kernelpcacut_modular(data, threshold, width):
	from shogun.Features import RealFeatures
	from shogun.Preprocessor import KernelPCACut
	from shogun.Kernel import GaussianKernel
	
	features = RealFeatures(data)
	
	kernel = GaussianKernel(features,features,width)
		
	preprocessor = KernelPCACut(kernel,threshold)
	preprocessor.init(features)
	preprocessor.apply_to_feature_matrix(features)

	return features


if __name__=='__main__':
	print 'KernelPCACut'
	preprocessor_kernelpcacut_modular(*parameter_list[0])

