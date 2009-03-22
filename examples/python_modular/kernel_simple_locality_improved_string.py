def simple_locality_improved_string ():
	print 'SimpleLocalityImprovedString'

	from shogun.Features import StringCharFeatures, DNA
	from shogun.Kernel import SimpleLocalityImprovedStringKernel

	feats_train=StringCharFeatures(fm_train_dna, DNA)
	feats_test=StringCharFeatures(fm_test_dna, DNA)
	length=5
	inner_degree=5
	outer_degree=7

	kernel=SimpleLocalityImprovedStringKernel(
		feats_train, feats_train, length, inner_degree, outer_degree)

	km_train=kernel.get_kernel_matrix()
	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()


if __name__=='__main__':
	from tools.load import LoadMatrix
	lm=LoadMatrix()
	fm_train_dna=lm.load_dna('../data/fm_train_dna.dat')
	fm_test_dna=lm.load_dna('../data/fm_test_dna.dat')
	simple_locality_improved_string()
