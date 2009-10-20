def combined():
	print 'Combined'
	from shogun.Kernel import CombinedKernel, GaussianKernel, FixedDegreeStringKernel, LocalAlignmentStringKernel
	from shogun.Features import RealFeatures, StringCharFeatures, CombinedFeatures, DNA

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

if __name__=='__main__':
	from tools.load import LoadMatrix
	from numpy import double
	lm=LoadMatrix()
	fm_train_real=double(lm.load_numbers('../data/fm_train_real.dat'))
	fm_test_real=double(lm.load_numbers('../data/fm_test_real.dat'))
	fm_train_dna=lm.load_dna('../data/fm_train_dna.dat')
	fm_test_dna=lm.load_dna('../data/fm_test_dna.dat')
	combined()


