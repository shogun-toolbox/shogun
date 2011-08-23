require 'modshogun'
require 'load'

traindat = load_numbers('../data/fm_train_word.dat')
testdat = load_numbers('../data/fm_test_word.dat')

parameter_list={{traindat,testdat,1.2},{traindat,testdat,1.2}}

function kernel_linear_word_modular (fm_train_word,fm_test_word,scale)
	feats_train=modshogun.WordFeatures(fm_train_word)
	feats_test=modshogun.WordFeatures(fm_test_word)

	kernel=modshogun.LinearKernel(feats_train, feats_train)
	kernel:set_normalizer(modshogun.AvgDiagKernelNormalizer(scale))
	kernel:init(feats_train, feats_train)

	km_train=kernel:get_kernel_matrix()
	kernel:init(feats_train, feats_test)
	km_test=kernel:get_kernel_matrix()
	return kernel
end

if debug.getinfo(3) == nill then
	print 'LinearWord'
	kernel_linear_word_modular(unpack(parameter_list[1]))
end
