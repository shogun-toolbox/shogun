def weighted_degree_position_string ():
	print 'WeightedDegreePositionString'
	from shogun.Features import StringCharFeatures, DNA
	from shogun.Kernel import WeightedDegreePositionStringKernel

	feats_train=StringCharFeatures(fm_train_dna, DNA)
	feats_test=StringCharFeatures(fm_test_dna, DNA)
	degree=20

	kernel=WeightedDegreePositionStringKernel(feats_train, feats_train, degree)

	from numpy import zeros,ones,float64,int32
	#kernel.set_shifts(zeros(len(fm_train_dna[0]), dtype=int32))
	#kernel.set_position_weights(ones(len(fm_train_dna[0]), dtype=float64))

	km_train=kernel.get_kernel_matrix()
	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()


if __name__=='__main__':
	from tools.load import LoadMatrix
	lm=LoadMatrix()
	fm_train_dna=lm.load_dna('../data/fm_train_dna.dat')
	fm_test_dna=lm.load_dna('../data/fm_test_dna.dat')
	weighted_degree_position_string()
