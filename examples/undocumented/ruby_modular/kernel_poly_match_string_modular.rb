# this was trancekoded by the awesome trancekoder
# ...and fixifikated by the awesum fixifikator
require 'modshogun'
require 'pp'
traindat = LoadMatrix.load_dna('../data/fm_train_dna.dat')
testdat = LoadMatrix.load_dna('../data/fm_test_dna.dat')

parameter_list = [[traindat,testdat,3,False],[traindat,testdat,4,False]]
# *** def kernel_poly_match_string_modular(fm_train_dna=traindat,fm_test_dna=testdat,degree=3,inhomogene=False)
def kernel_poly_match_string_modular(fm_train_dna=traindat,fm_test_dna=testdat,degree=3,inhomogene=Modshogun::False.new
def kernel_poly_match_string_modular(fm_train_dna=traindat,fm_test_dna=testdat,degree=3,inhomogene.set_features)

# *** 	feats_train=StringCharFeatures(fm_train_dna, DNA)
	feats_train=Modshogun::StringCharFeatures.new
	feats_train.set_features(fm_train_dna, DNA)
# *** 	feats_test=StringCharFeatures(fm_train_dna, DNA)
	feats_test=Modshogun::StringCharFeatures.new
	feats_test.set_features(fm_train_dna, DNA)

# *** 	kernel=PolyMatchStringKernel(feats_train, feats_train, degree, inhomogene)
	kernel=Modshogun::PolyMatchStringKernel.new
	kernel.set_features(feats_train, feats_train, degree, inhomogene)

	km_train=kernel.get_kernel_matrix()
	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	return km_train,km_test,kernel


end
if __FILE__ == $0
	puts 'PolyMatchString'
	kernel_poly_match_string_modular(*parameter_list[0])

end
