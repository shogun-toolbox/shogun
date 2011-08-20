# this was trancekoded by the awesome trancekoder
# ...and fixifikated by the awesum fixifikator
require 'modshogun'
require 'pp'
traindat = LoadMatrix.load_dna('../data/fm_train_dna.dat')
testdat = LoadMatrix.load_dna('../data/fm_test_dna.dat')

parameter_list = [[traindat,testdat,3,1.2,10],[traindat,testdat,4,1.3,10]]

def kernel_oligo_string_modular(fm_train_dna=traindat,fm_test_dna=testdat,k=3,width=1.2,size_cache=10)

# *** 	feats_train=StringCharFeatures(fm_train_dna, DNA)
	feats_train=Modshogun::StringCharFeatures.new
	feats_train.set_features(fm_train_dna, DNA)
# *** 	feats_test=StringCharFeatures(fm_test_dna, DNA)
	feats_test=Modshogun::StringCharFeatures.new
	feats_test.set_features(fm_test_dna, DNA)

# *** 	kernel=OligoStringKernel(size_cache, k, width)
	kernel=Modshogun::OligoStringKernel.new
	kernel.set_features(size_cache, k, width)
	kernel.init(feats_train, feats_train)

	km_train=kernel.get_kernel_matrix()
	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	return km_train,km_test,kernel


end
if __FILE__ == $0
	puts 'OligoString'
	kernel_oligo_string_modular(*parameter_list[0])

end
