def weighted_degree_string ():
	print 'WeightedDegreeString'
	from shogun.Features import StringCharFeatures, DNA
	from shogun.Kernel import WeightedDegreeStringKernel

	feats_train=StringCharFeatures(fm_train_dna, DNA)
	feats_test=StringCharFeatures(fm_test_dna, DNA)
	degree=20

	kernel=WeightedDegreeStringKernel(feats_train, feats_train, degree)

	#weights=arange(1,degree+1,dtype=double)[::-1]/ \
	#	sum(arange(1,degree+1,dtype=double))
	#kernel.set_wd_weights(weights)

	km_train=kernel.get_kernel_matrix()
	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()


if __name__=='__main__':
	from tools.load import LoadMatrix
	lm=LoadMatrix()
	fm_train_dna=lm.load_dna('../data/fm_train_dna.dat')
	fm_test_dna=lm.load_dna('../data/fm_test_dna.dat')
	weighted_degree_string()
