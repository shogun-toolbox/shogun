# this was trancekoded by the awesome trancekoder
# ...and fixifikated by the awesum fixifikator
require 'modshogun'
require 'pp'

traindat = LoadMatrix.load_dna('../data/fm_train_dna.dat')
testdat = LoadMatrix.load_dna('../data/fm_test_dna.dat')
parameter_list=[[traindat, testdat,3],[traindat,testdat,4]]

def kernel_fixed_degree_string_modular(fm_train_dna=traindat, fm_test_dna=testdat,degree=3)

# *** 	feats_train=StringCharFeatures(fm_train_dna, DNA)
	feats_train=Modshogun::StringCharFeatures.new
	feats_train.set_features(fm_train_dna, DNA)
# *** 	feats_test=StringCharFeatures(fm_test_dna, DNA)
	feats_test=Modshogun::StringCharFeatures.new
	feats_test.set_features(fm_test_dna, DNA)

# *** 	kernel=FixedDegreeStringKernel(feats_train, feats_train, degree)
	kernel=Modshogun::FixedDegreeStringKernel.new
	kernel.set_features(feats_train, feats_train, degree)

	km_train=kernel.get_kernel_matrix()
	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()

	return km_train,km_test,kernel

end
if __FILE__ == $0
	puts 'FixedDegreeString'
	kernel_fixed_degree_string_modular(*parameter_list[0])

end
