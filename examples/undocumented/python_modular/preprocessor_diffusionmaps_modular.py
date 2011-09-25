from tools.load import LoadMatrix

lm=LoadMatrix()
data = lm.load_numbers('../data/fm_train_real.dat')

parameter_list = [[data,10],[data,20]]

def preprocessor_diffusionmaps_modular(data,t):
	from shogun.Features import RealFeatures
	from shogun.Preprocessor import DiffusionMaps
	from shogun.Kernel import GaussianKernel
	
	features = RealFeatures(data)
		
	preprocessor = DiffusionMaps()
	preprocessor.set_target_dim(1)
	preprocessor.set_kernel(GaussianKernel(10,10.0))
	preprocessor.set_t(t)
	preprocessor.apply_to_feature_matrix(features)

	return features


if __name__=='__main__':
	print 'DiffusionMaps'
	preprocessor_diffusionmaps_modular(*parameter_list[0])

