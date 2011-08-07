require 'narray'
require 'modshogun'
require 'load'
require 'pp'

traindat = LoadMatrix.load_numbers('../data/fm_train_byte.dat')
testdat = LoadMatrix.load_numbers('../data/fm_test_byte.dat')

parameter_list=[[traindat,testdat],[traindat,testdat]]

def kernel_linear_byte_modular(fm_train_byte=traindat,fm_test_byte=testdat)
	pp fm_train_byte
	feats_train=Modshogun::ByteFeatures.new(fm_train_byte)
	feats_test=Modshogun::ByteFeatures.new(fm_test_byte)

	kernel=Modshogun::LinearKernel.new(feats_train, feats_train)
	km_train=kernel.get_kernel_matrix()

	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	pp " km_train", km_train
	pp " km_test", km_test
	return kernel
end

if __FILE__ == $0
	puts 'LinearByte'
	kernel_linear_byte_modular(*parameter_list[0])
end
