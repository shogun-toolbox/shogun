from tools.load import LoadMatrix

lm=LoadMatrix()
data = lm.load_numbers('../data/fm_train_real.dat')

parameter_list = [[data,10],[data,20]]

def converter_diffusionmaps_modular(data,t):
	from shogun.Features import RealFeatures
	from shogun.Converter import DiffusionMaps
	from shogun.Kernel import GaussianKernel
	
	features = RealFeatures(data)
		
	converter = DiffusionMaps()
	converter.set_target_dim(1)
	converter.set_kernel(GaussianKernel(10,10.0))
	converter.set_t(t)
	converter.apply(features)

	return features


if __name__=='__main__':
	print 'DiffusionMaps'
	converter_diffusionmaps_modular(*parameter_list[0])

