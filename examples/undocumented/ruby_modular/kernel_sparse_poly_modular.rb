# this was trancekoded by the awesome trancekoder
require 'narray'
require 'modshogun'
require 'load'
require 'pp'
traindat = LoadMatrix.load_numbers('../data/fm_train_real.dat')
testdat = LoadMatrix.load_numbers('../data/fm_test_real.dat')

parameter_list = [[traindat,testdat,10,3,True],[traindat,testdat,10,4,True]]

def kernel_sparse_poly_modular(fm_train_real=traindat,fm_test_real=testdat,
		 size_cache=10,degree=3,inhomogene=True ):


	feats_train=SparseRealFeatures(fm_train_real)
	feats_test=SparseRealFeatures(fm_test_real)



	kernel=PolyKernel(feats_train, feats_train, size_cache, degree,
		inhomogene)
	km_train=kernel.get_kernel_matrix()

	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	return km_train,km_test,kernel


end
if __FILE__ == $0
	print 'SparsePoly'
	kernel_sparse_poly_modular(*parameter_list[0])

end
