require 'modshogun'
require 'pp'

traindat = LoadMatrix.load_numbers('../data/fm_train_real.dat')
testdat = LoadMatrix.load_numbers('../data/fm_test_real.dat')

parameter_list=[[traindat,testdat, 1.0],[traindat,testdat, 10.0]]

def kernel_cauchy_modular(fm_train_real=traindat,fm_test_real=testdat, sigma=1.0)

# *** 	feats_train=RealFeatures(fm_train_real)
	feats_train=Modshogun::RealFeatures.new
	feats_train.set_feature_matrix(fm_train_real)
# *** 	feats_test=RealFeatures(fm_test_real)
	feats_test=Modshogun::RealFeatures.new
	feats_test.set_feature_matrix(fm_test_real)
	
# *** 	distance=EuclidianDistance(feats_train, feats_train)
	distance=Modshogun::EuclidianDistance.new(feats_train, feats_train)

# *** 	kernel=CauchyKernel(feats_train, feats_train, sigma, distance)
	kernel=Modshogun::CauchyKernel.new(feats_train, feats_train, sigma, distance)
	km_train=kernel.get_kernel_matrix()

	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	return km_train,km_test,kernel
end

if __FILE__ == $0
	puts 'Cauchy'
	pp kernel_cauchy_modular(*parameter_list[0])
end
