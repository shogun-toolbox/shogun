# this was trancekoded by the awesome trancekoder
# ...and fixifikated by the awesum fixifikator
require 'modshogun'
require 'pp'

traindat = LoadMatrix.load_numbers('../data/fm_train_real.dat')
testdat = LoadMatrix.load_numbers('../data/fm_test_real.dat')

parameter_list=[[traindat,testdat, 1.0],[traindat,testdat, 5.0]]

def kernel_exponential_modular(fm_train_real=traindat,fm_test_real=testdat, tau_coef=1.0)

# *** 	feats_train=RealFeatures(fm_train_real)
	feats_train=Modshogun::RealFeatures.new
	feats_train.set_features(fm_train_real)
# *** 	feats_test=RealFeatures(fm_test_real)
	feats_test=Modshogun::RealFeatures.new
	feats_test.set_features(fm_test_real)

	distance = EuclidianDistance(feats_train, feats_train)
	
# *** 	kernel=ExponentialKernel(feats_train, feats_train, tau_coef, distance, 10)
	kernel=Modshogun::ExponentialKernel.new
	kernel.set_features(feats_train, feats_train, tau_coef, distance, 10)
	km_train=kernel.get_kernel_matrix()

	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	return km_train,km_test,kernel



end
if __FILE__ == $0
	puts 'Exponential'
	kernel_exponential_modular(*parameter_list[0])

end
