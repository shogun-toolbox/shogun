# this was trancekoded by the awesome trancekoder
require 'narray'
require 'modshogun'
require 'load'
require 'pp'
###########################################################################
# chi2 kernel
###########################################################################

traindat = double(LoadMatrix.load_numbers('../data/fm_train_real.dat'))
testdat = double(LoadMatrix.load_numbers('../data/fm_test_real.dat'))
parameter_list = [[traindat,testdat,1.4,10], [traindat,testdat,1.5,10]]

def kernel_chi2_modular(fm_train_real=traindat,fm_test_real=testdat,width=1.4, size_cache=10)
	
	feats_train=RealFeatures(fm_train_real)
	feats_test=RealFeatures(fm_test_real)
	
	
	kernel=Chi2Kernel(feats_train, feats_train, width, size_cache)

	km_train=kernel.get_kernel_matrix()
	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	return km_train,km_test,kernel


end
if __FILE__ == $0
	print 'Chi2'
	kernel_chi2_modular(*parameter_list[0])

end
