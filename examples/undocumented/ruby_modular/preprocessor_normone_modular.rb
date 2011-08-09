# this was trancekoded by the awesome trancekoder
require 'narray'
require 'modshogun'
require 'load'
require 'pp'

traindat = LoadMatrix.load_numbers('../data/fm_train_real.dat')
testdat = LoadMatrix.load_numbers('../data/fm_test_real.dat')

parameter_list = [[traindat,testdat,1.4,10],[traindat,testdat,1.5,10]]

def preprocessor_normone_modular(fm_train_real=traindat,fm_test_real=testdat,width=1.4,size_cache=10)


	feats_train=RealFeatures(fm_train_real)
	feats_test=RealFeatures(fm_test_real)

	preprocessor=NormOne()
	preprocessor.init(feats_train)
	feats_train.add_preprocessor(preprocessor)
	feats_train.apply_preprocessor()
	feats_test.add_preprocessor(preprocessor)
	feats_test.apply_preprocessor()

	kernel=Chi2Kernel(feats_train, feats_train, width, size_cache)

	km_train=kernel.get_kernel_matrix()
	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()

	return km_train,km_test,kernel


end
if __FILE__ == $0
	print 'NormOne'
	preprocessor_normone_modular(*parameter_list[0])

end
