# this was trancekoded by the awesome trancekoder
require 'narray'
require 'modshogun'
require 'load'
require 'pp'
traindat = LoadMatrix.load_dna('../data/fm_train_dna.dat')
testdat = LoadMatrix.load_dna('../data/fm_test_dna.dat')
label_traindat = LoadMatrix.load_labels('../data/label_train_dna.dat')

parameter_list = [[traindat,testdat,label_traindat,3,0,False],[traindat,testdat,label_traindat,3,0,False]]
def kernel_salzberg_word_string_modular(fm_train_dna=traindat,fm_test_dna=testdat,label_train_dna=label_traindat,

end
order=3,gap=0,reverse=False):

	charfeat=StringCharFeatures(fm_train_dna, DNA)
	feats_train=StringWordFeatures(charfeat.get_alphabet())
	feats_train.obtain_from_char(charfeat, order-1, order, gap, reverse)

	charfeat=StringCharFeatures(fm_test_dna, DNA)
	feats_test=StringWordFeatures(charfeat.get_alphabet())
	feats_test.obtain_from_char(charfeat, order-1, order, gap, reverse)

	pie=PluginEstimate()
	labels=Labels(label_train_dna)
	pie.set_labels(labels)
	pie.set_features(feats_train)
	pie.train()

	kernel=SalzbergWordStringKernel(feats_train, feats_train, pie, labels)
	km_train=kernel.get_kernel_matrix()

	kernel.init(feats_train, feats_test)
	pie.set_features(feats_test)
	pie.apply().get_labels()
	km_test=kernel.get_kernel_matrix()
	return km_train,km_test,kernel

if __FILE__ == $0
	print 'PluginEstimate w/ SalzbergWord'
	kernel_salzberg_word_string_modular(*parameter_list[0])


end
