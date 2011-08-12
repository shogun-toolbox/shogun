# this was trancekoded by the awesome trancekoder
require 'narray'
require 'modshogun'
require 'load'
require 'pp'

data = LoadMatrix.load_numbers('../data/fm_train_real.dat')

parameter_list = [[data, 0.01, 1.0], [data, 0.05, 2.0]]

def preprocessor_kernelpcacut_modular(data, threshold, width)
	
	features = RealFeatures(data)
	
	kernel = GaussianKernel(features,features,width)
		
	preprocessor = KernelPCACut(kernel,threshold)
	preprocessor.init(features)
	preprocessor.apply_to_feature_matrix(features)

	return features



end
if __FILE__ == $0
	print 'KernelPCACut'
	preprocessor_kernelpcacut_modular(*parameter_list[0])


end
