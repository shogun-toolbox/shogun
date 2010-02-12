def oligo_string ():
	print 'OligoString'
	from shogun.Features import StringCharFeatures, DNA
	from shogun.Kernel import OligoStringKernel

	feats_train=StringCharFeatures(fm_train_dna, DNA)
	feats_test=StringCharFeatures(fm_test_dna, DNA)
	k=3
	width=1.2
	size_cache=10

	kernel=OligoStringKernel(size_cache, k, width)
	kernel.init(feats_train, feats_train)

	km_train=kernel.get_kernel_matrix()
	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()

if __name__=='__main__':
	from tools.load import LoadMatrix
	lm=LoadMatrix()
	fm_train_dna=lm.load_dna('../data/fm_train_dna.dat')
	fm_test_dna=lm.load_dna('../data/fm_test_dna.dat')
	oligo_string()
