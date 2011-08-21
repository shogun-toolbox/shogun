# this was trancekoded by the awesome trancekoder
# ...and fixifikated by the awesum fixifikator
require 'modshogun'
require 'pp'
traindat = LoadMatrix.load_dna('../data/fm_train_dna.dat')
testdat = LoadMatrix.load_dna('../data/fm_test_dna.dat')

parameter_list = [[traindat,testdat],[traindat,testdat]]

# *** def kernel_weighted_comm_word_string_modular(fm_train_dna=traindat,fm_test_dna=testdat,order=3,gap=0,reverse=True )
def kernel_weighted_comm_word_string_modular(fm_train_dna=traindat,fm_test_dna=testdat,order=3,gap=0,reverse=Modshogun::True.new
def kernel_weighted_comm_word_string_modular(fm_train_dna=traindat,fm_test_dna=testdat,order=3,gap=0,reverse.set_features )

# *** 	charfeat=StringCharFeatures(fm_train_dna, DNA)
	charfeat=Modshogun::StringCharFeatures.new
	charfeat.set_features(fm_train_dna, DNA)
# *** 	feats_train=StringWordFeatures(charfeat.get_alphabet())
	feats_train=Modshogun::StringWordFeatures.new
	feats_train.set_features(charfeat.get_alphabet())
	feats_train.obtain_from_char(charfeat, order-1, order, gap, reverse)
# *** 	preproc=SortWordString()
	preproc=Modshogun::SortWordString.new
	preproc.set_features()
	preproc.init(feats_train)
	feats_train.add_preprocessor(preproc)
	feats_train.apply_preprocessor()

# *** 	charfeat=StringCharFeatures(fm_test_dna, DNA)
	charfeat=Modshogun::StringCharFeatures.new
	charfeat.set_features(fm_test_dna, DNA)
# *** 	feats_test=StringWordFeatures(charfeat.get_alphabet())
	feats_test=Modshogun::StringWordFeatures.new
	feats_test.set_features(charfeat.get_alphabet())
	feats_test.obtain_from_char(charfeat, order-1, order, gap, reverse)
	feats_test.add_preprocessor(preproc)
	feats_test.apply_preprocessor()

# *** 	use_sign=False
	use_sign=Modshogun::False.new
	use_sign.set_features
# *** 	kernel=WeightedCommWordStringKernel(feats_train, feats_train, use_sign)
	kernel=Modshogun::WeightedCommWordStringKernel.new
	kernel.set_features(feats_train, feats_train, use_sign)
	km_train=kernel.get_kernel_matrix()

	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	return km_train,km_test,kernel


end
if __FILE__ == $0
	puts 'WeightedCommWordString'
	kernel_weighted_comm_word_string_modular(*parameter_list[0])

end
