# this was trancekoded by the awesome trancekoder
# ...and fixifikated by the awesum fixifikator
require 'modshogun'
require 'pp'

traindat = LoadMatrix.load_numbers('../data/fm_train_real.dat')
testdat = LoadMatrix.load_numbers('../data/fm_test_real.dat')

parameter_list=[[traindat,testdat, 1.5, 1.0],[traindat,testdat, 1.0, 1.5]]

def kernel_wavelet_modular(fm_train_real=traindat,fm_test_real=testdat, dilation=1.5, translation=1.0)

# *** 	feats_train=RealFeatures(fm_train_real)
	feats_train=Modshogun::RealFeatures.new
	feats_train.set_features(fm_train_real)
# *** 	feats_test=RealFeatures(fm_test_real)
	feats_test=Modshogun::RealFeatures.new
	feats_test.set_features(fm_test_real)

# *** 	kernel=WaveletKernel(feats_train, feats_train, 10, dilation, translation)
	kernel=Modshogun::WaveletKernel.new
	kernel.set_features(feats_train, feats_train, 10, dilation, translation)
	km_train=kernel.get_kernel_matrix()

	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	return km_train,km_test,kernel



end
if __FILE__ == $0
	puts 'Wavelet'
	kernel_wavelet_modular(*parameter_list[0])

end
