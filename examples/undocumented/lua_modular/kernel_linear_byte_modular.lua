require 'modshogun'
require 'load'

traindat = load_numbers('../data/fm_train_byte.dat')
testdat = load_numbers('../data/fm_test_byte.dat')

parameter_list={{traindat,testdat},{traindat,testdat}}

function kernel_linear_byte_modular(fm_train_byte,fm_test_byte)
	feats_train=modshogun.ByteFeatures(fm_train_byte)
	feats_test=modshogun.ByteFeatures(fm_test_byte)

	kernel=modshogun.LinearKernel(feats_train, feats_train)
	km_train=kernel:get_kernel_matrix()

	kernel:init(feats_train, feats_test)
	km_test=kernel:get_kernel_matrix()
	return kernel
end

if debug.getinfo(3) == nill then
	print 'LinearByte'
	kernel_linear_byte_modular(unpack(parameter_list[1]))
end
