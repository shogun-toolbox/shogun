# this was trancekoded by the awesome trancekoder
# ...and fixifikated by the awesum fixifikator
require 'modshogun'
require 'pp'

data = LoadMatrix.load_numbers('../data/fm_train_real.dat')

parameter_list = [[data, 0.01, 1.0], [data, 0.05, 2.0]]

def preprocessor_kernelpca_modular(data, threshold, width)
	
	features = RealFeatures(data)
	
	kernel = GaussianKernel(features,features,width)
		
	preprocessor = KernelPCA(kernel)
	preprocessor.init(features)
	preprocessor.apply_to_feature_matrix(features)

	return features



end
if __FILE__ == $0
	print 'KernelPCA'
	preprocessor_kernelpca_modular(*parameter_list[0])

end
