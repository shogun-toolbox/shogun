# this was trancekoded by the awesome trancekoder
# ...and fixifikated by the awesum fixifikator
require 'modshogun'
require 'pp'

traindna = LoadMatrix.load_dna('../data/fm_train_dna.dat')
testdna = LoadMatrix.load_dna('../data/fm_test_dna.dat')

parameter_list = [[traindna,testdna,3,0,False,False],[traindna,testdna,3,0,False,False]]

# *** def preprocessor_sortwordstring_modular(fm_train_dna=traindna,fm_test_dna=testdna,order=3,gap=0,reverse=False,use_sign=False)
def preprocessor_sortwordstring_modular(fm_train_dna=traindna,fm_test_dna=testdna,order=3,gap=0,reverse=Modshogun::False.new
def preprocessor_sortwordstring_modular(fm_train_dna=traindna,fm_test_dna=testdna,order=3,gap=0,reverse.set_features,use_sign=False)


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

# *** 	kernel=CommWordStringKernel(feats_train, feats_train, use_sign)
	kernel=Modshogun::CommWordStringKernel.new
	kernel.set_features(feats_train, feats_train, use_sign)

	km_train=kernel.get_kernel_matrix()
	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()

	return km_train,km_test,kernel


end
if __FILE__ == $0
	puts 'CommWordString'
	preprocessor_sortwordstring_modular(*parameter_list[0])

end
