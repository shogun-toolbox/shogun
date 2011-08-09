# this was trancekoded by the awesome trancekoder
require 'narray'
require 'modshogun'
require 'load'
require 'pp'

traindat = LoadMatrix.load_numbers('../data/fm_train_real.dat')
testdat = LoadMatrix.load_numbers('../data/fm_test_real.dat')

parameter_list = [[traindat,testdat,10,1.2,1.3],[traindat,testdat,10,1.2,1.3]]

def kernel_sigmoid_modular(fm_train_real=traindat,fm_test_real=testdat,size_cache=10,gamma=1.2,coef0=1.3)


	feats_train=RealFeatures(fm_train_real)
	feats_test=RealFeatures(fm_test_real)
	

	kernel=SigmoidKernel(feats_train, feats_train, size_cache, gamma, coef0)
	km_train=kernel.get_kernel_matrix()

	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	return km_train,km_test,kernel
	

end
if __FILE__ == $0
	print 'Sigmoid'
	kernel_sigmoid_modular(*parameter_list[0])

end
