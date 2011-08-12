# this was trancekoded by the awesome trancekoder
require 'narray'
require 'modshogun'
require 'load'
require 'pp'

traindat = double(LoadMatrix.load_numbers('../data/fm_train_real.dat'))
testdat = double(LoadMatrix.load_numbers('../data/fm_test_real.dat'))
traindna = LoadMatrix.load_dna('../data/fm_train_dna.dat')
testdna = LoadMatrix.load_dna('../data/fm_test_dna.dat')
 
parameter_list = [[traindat,testdat,traindna,testdna],[traindat,testdat,traindna,testdna]]
def kernel_combined_modular(fm_train_real=traindat,fm_test_real=testdat,fm_train_dna=traindna,fm_test_dna=testdna )

	kernel=CombinedKernel()
	feats_train=CombinedFeatures()
	feats_test=CombinedFeatures()

	subkfeats_train=RealFeatures(fm_train_real)
	subkfeats_test=RealFeatures(fm_test_real)
	subkernel=GaussianKernel(10, 1.1)
	feats_train.append_feature_obj(subkfeats_train)
	feats_test.append_feature_obj(subkfeats_test)
	kernel.append_kernel(subkernel)

	subkfeats_train=StringCharFeatures(fm_train_dna, DNA)
	subkfeats_test=StringCharFeatures(fm_test_dna, DNA)
	degree=3
	subkernel=FixedDegreeStringKernel(10, degree)
	feats_train.append_feature_obj(subkfeats_train)
	feats_test.append_feature_obj(subkfeats_test)
	kernel.append_kernel(subkernel)

	subkfeats_train=StringCharFeatures(fm_train_dna, DNA)
	subkfeats_test=StringCharFeatures(fm_test_dna, DNA)
	subkernel=LocalAlignmentStringKernel(10)
	feats_train.append_feature_obj(subkfeats_train)
	feats_test.append_feature_obj(subkfeats_test)
	kernel.append_kernel(subkernel)

	kernel.init(feats_train, feats_train)
	km_train=kernel.get_kernel_matrix()
	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	return km_train,km_test,kernel


end
if __FILE__ == $0
	print 'Combined'
	kernel_combined_modular(*parameter_list[0])



end
