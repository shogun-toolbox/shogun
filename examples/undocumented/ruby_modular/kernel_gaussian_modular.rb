require 'narray'
require 'modshogun'
require 'load'
require 'pp'

traindat = LoadMatrix.load_numbers('../data/fm_train_real.dat')
testdat = LoadMatrix.load_numbers('../data/fm_test_real.dat')

parameter_list=[[traindat,testdat, 1.3],[traindat,testdat, 1.4]]

def kernel_gaussian_modular(fm_train_real=traindat,fm_test_real=testdat, width=1.3)
  pp fm_train_real
	feats_train=Modshogun::RealFeatures.new
	feats_train.set_feature_matrix fm_train_real
	feats_test=Modshogun::RealFeatures.new
	feats_test.set_feature_matrix fm_test_real

	kernel=Modshogun::GaussianKernel.new feats_train, feats_train, width
	km_train=kernel.get_kernel_matrix()

	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	pp km_train
	return km_train,km_test,kernel
end

if __FILE__ == $0 then
	puts 'Gaussian'
	kernel_gaussian_modular(*parameter_list[0])
end
