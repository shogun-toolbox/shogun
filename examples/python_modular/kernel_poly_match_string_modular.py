def poly_match_string ():
	print 'PolyMatchString'
	from shogun.Kernel import PolyMatchStringKernel
	from shogun.Features import StringCharFeatures, DNA

	feats_train=StringCharFeatures(fm_train_dna, DNA)
	feats_test=StringCharFeatures(fm_train_dna, DNA)
	degree=3
	inhomogene=False

	kernel=PolyMatchStringKernel(feats_train, feats_train, degree, inhomogene)

	km_train=kernel.get_kernel_matrix()
	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()


if __name__=='__main__':
	from tools.load import LoadMatrix
	lm=LoadMatrix()
	fm_train_dna=lm.load_dna('../data/fm_train_dna.dat')
	fm_test_dna=lm.load_dna('../data/fm_test_dna.dat')
	poly_match_string()
