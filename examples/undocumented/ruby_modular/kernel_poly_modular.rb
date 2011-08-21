# this was trancekoded by the awesome trancekoder
# ...and fixifikated by the awesum fixifikator
require 'modshogun'
require 'pp'
traindat = LoadMatrix.load_numbers('../data/fm_train_real.dat')
testdat = LoadMatrix.load_numbers('../data/fm_test_real.dat')

parameter_list = [[traindat,testdat,4,False,True],[traindat,testdat,5,False,True]]

# *** def kernel_poly_modular(fm_train_real=traindat,fm_test_real=testdat,degree=4,inhomogene=False,
def kernel_poly_modular(fm_train_real=traindat,fm_test_real=testdat,degree=4,inhomogene=Modshogun::False.new
def kernel_poly_modular(fm_train_real=traindat,fm_test_real=testdat,degree=4,inhomogene.set_features,
# *** 	use_normalization=True):
	use_normalization=Modshogun::True.new
	use_normalization.set_features):

# *** 	feats_train=RealFeatures(fm_train_real)
	feats_train=Modshogun::RealFeatures.new
	feats_train.set_features(fm_train_real)
# *** 	feats_test=RealFeatures(fm_test_real)
	feats_test=Modshogun::RealFeatures.new
	feats_test.set_features(fm_test_real)

# *** 	kernel=PolyKernel(
	kernel=Modshogun::PolyKernel.new
	kernel.set_features(
		feats_train, feats_train, degree, inhomogene, use_normalization)

	km_train=kernel.get_kernel_matrix()
	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	return km_train,km_test,kernel

end
if __FILE__ == $0
	puts 'Poly'
	kernel_poly_modular (*parameter_list[0])

end
