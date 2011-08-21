# this was trancekoded by the awesome trancekoder
# ...and fixifikated by the awesum fixifikator
require 'modshogun'
require 'pp'
traindat = LoadMatrix.load_dna('../data/fm_train_dna.dat')
testdat = LoadMatrix.load_dna('../data/fm_test_dna.dat')

parameter_list = [[traindat,testdat,2,True,3,0,False],[traindat,testdat,2,True,3,0,False]]

def kernel_poly_match_word_string_modular(fm_train_dna=traindat,fm_test_dna=testdat,

end
# *** degree=2,inhomogene=True,order=3,gap=0,reverse=False):
degree=2,inhomogene=Modshogun::True.new
degree=2,inhomogene.set_features,order=3,gap=0,reverse=False):



# *** 	charfeat=StringCharFeatures(fm_train_dna, DNA)
	charfeat=Modshogun::StringCharFeatures.new
	charfeat.set_features(fm_train_dna, DNA)
# *** 	feats_train=StringWordFeatures(DNA)
	feats_train=Modshogun::StringWordFeatures.new
	feats_train.set_features(DNA)
	feats_train.obtain_from_char(charfeat, order-1, order, gap, reverse)

# *** 	charfeat=StringCharFeatures(fm_test_dna, DNA)
	charfeat=Modshogun::StringCharFeatures.new
	charfeat.set_features(fm_test_dna, DNA)
# *** 	feats_test=StringWordFeatures(DNA)
	feats_test=Modshogun::StringWordFeatures.new
	feats_test.set_features(DNA)
	feats_test.obtain_from_char(charfeat, order-1, order, gap, reverse)

# *** 	kernel=PolyMatchWordStringKernel(feats_train, feats_train, degree, inhomogene)
	kernel=Modshogun::PolyMatchWordStringKernel.new
	kernel.set_features(feats_train, feats_train, degree, inhomogene)

	km_train=kernel.get_kernel_matrix()
	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	return km_train,km_test,kernel

if __FILE__ == $0
	puts 'PolyMatchWordString'
	kernel_poly_match_word_string_modular(*parameter_list[0])

end
