require 'nmatrix'
require 'modshogun'
require 'pp'

require_relative 'load'

traindat = LoadMatrix.load_numbers('../data/fm_train_real.dat')
testdat = LoadMatrix.load_numbers('../data/fm_test_real.dat')

parameter_list=[[traindat,testdat, 1.0],[traindat,testdat, 5.0]]

def kernel_circular_modular(fm_train_real=traindat,fm_test_real=testdat, sigma=1.0)

	feats_train=Modshogun::RealFeatures.new
	feats_train.set_feature_matrix(fm_train_real)
	feats_test=Modshogun::RealFeatures.new
	feats_test.set_feature_matrix(fm_test_real)
	
	distance=Modshogun::EuclideanDistance.new(feats_train, feats_train)

	kernel=Modshogun::CircularKernel.new(feats_train, feats_train, sigma, distance)
	km_train=kernel.get_kernel_matrix()

	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	return km_train,km_test,kernel
end

if __FILE__ == $0
	puts 'Circular'
	pp kernel_circular_modular(*parameter_list[0])
end
