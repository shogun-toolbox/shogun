require 'modshogun'
require 'load'

traindat = load_numbers('../data/fm_train_real.dat')
testdat = load_numbers('../data/fm_test_real.dat')

parameter_list = {{traindat,testdat, 1.3},{traindat,testdat, 1.4}}

function kernel_gaussian_modular (fm_train_real,fm_test_real,width)

	feats_train=modshogun.RealFeatures(fm_train_real)
	feats_test=modshogun.RealFeatures(fm_test_real)

	kernel=modshogun.GaussianKernel(feats_train, feats_train, width)

	km_train=kernel:get_kernel_matrix()
	kernel:init(feats_train, feats_test)
	km_test=kernel:get_kernel_matrix()

	return km_train,km_test,kernel
end

if debug.getinfo(3) == nill then
	print 'Gaussian'
	kernel_gaussian_modular(unpack(parameter_list[1]))
end
