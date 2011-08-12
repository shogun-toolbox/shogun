require 'narray'
require 'modshogun'
require 'load'
require 'pp'

traindat = LoadMatrix.load_numbers('../data/fm_train_word.dat')
testdat = LoadMatrix.load_numbers('../data/fm_test_word.dat')

parameter_list=[[traindat,testdat,1.2],[traindat,testdat,1.2]]

def kernel_linear_word_modular(fm_train_word=traindat,fm_test_word=testdat,scale=1.2)
	feats_train=Modshogun::WordFeatures.new(fm_train_word)
	feats_test=Modshogun::WordFeatures.new(fm_test_word)

	kernel=Modshogun::LinearKernel.new(feats_train, feats_train)
	kernel.set_normalizer(Modshogun::AvgDiagKernelNormalizer.new(scale))
	kernel.init(feats_train, feats_train)

	km_train=kernel.get_kernel_matrix()
	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	pp " km_train", km_train
	pp " km_test", km_test
	return kernel
end

if __FILE__ == $0
	puts 'LinearWord'
	kernel_linear_word_modular(*parameter_list[0])
end
