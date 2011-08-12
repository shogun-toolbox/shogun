# this was trancekoded by the awesome trancekoder
require 'narray'
require 'modshogun'
require 'load'
require 'pp'

traindat = LoadMatrix.load_numbers('../data/fm_train_real.dat')
testdat = LoadMatrix.load_numbers('../data/fm_test_real.dat')

parameter_list = [[traindat,testdat,1.1],[traindat,testdat,1.2]]

def kernel_sparse_linear_modular(fm_train_real=traindat,fm_test_real=testdat,scale=1.1)

	feats_train=SparseRealFeatures(fm_train_real)
	feats_test=SparseRealFeatures(fm_test_real)

	kernel=LinearKernel()
	kernel.set_normalizer(AvgDiagKernelNormalizer(scale))
	kernel.init(feats_train, feats_train)
	km_train=kernel.get_kernel_matrix()

	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	return km_train,km_test,kernel


end
if __FILE__ == $0
	print 'SparseLinear'
	kernel_sparse_linear_modular(*parameter_list[0])

end
