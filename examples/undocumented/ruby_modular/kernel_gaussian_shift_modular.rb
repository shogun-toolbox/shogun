# this was trancekoded by the awesome trancekoder
# ...and fixifikated by the awesum fixifikator
require 'modshogun'
require 'pp'

traindat = LoadMatrix.load_numbers('../data/fm_train_real.dat')
testdat = LoadMatrix.load_numbers('../data/fm_test_real.dat')

parameter_list=[[traindat,testdat,1.8,2,1],[traindat,testdat,1.9,2,1]]

def kernel_gaussian_shift_modular(fm_train_real=traindat,fm_test_real=testdat,width=1.8,max_shift=2,shift_step=1)

# *** 	feats_train=RealFeatures(fm_train_real)
	feats_train=Modshogun::RealFeatures.new
	feats_train.set_features(fm_train_real)
# *** 	feats_test=RealFeatures(fm_test_real)
	feats_test=Modshogun::RealFeatures.new
	feats_test.set_features(fm_test_real)

# *** 	kernel=GaussianShiftKernel(feats_train, feats_train, width, max_shift, shift_step)
	kernel=Modshogun::GaussianShiftKernel.new
	kernel.set_features(feats_train, feats_train, width, max_shift, shift_step)

	km_train=kernel.get_kernel_matrix()
	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()

	return km_train,km_test,kernel


end
if __FILE__ == $0
	puts 'GaussianShift'
	kernel_gaussian_shift_modular(*parameter_list[0])

end
