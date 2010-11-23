from tools.load import LoadMatrix
lm=LoadMatrix()
parameter_list=[[lm.load_numbers('../data/fm_train_dna.dat'),lm.load_numbers('../data/fm_test_dna.dat')],[lm.load_numbers('../data/fm_train_dna.dat'),lm.load_numbers('../data/fm_test_dna.dat')]]

def kernel_linear_string_modular (fm_train_real=lm.load_numbers('../data/fm_train_dna.dat'),fm_test_real=lm.load_numbers('../data/fm_test_dna.dat')):
	print 'LinearString'
	from shogun.Features import StringCharFeatures, DNA
	from shogun.Kernel import LinearStringKernel
	fm_train_dna=fm_train_dna
	fm_test_dna = fm_test_dna

	feats_train=StringCharFeatures(fm_train_dna, DNA)
	feats_test=StringCharFeatures(fm_test_dna, DNA)

	kernel=LinearStringKernel(feats_train, feats_train)
	km_train=kernel.get_kernel_matrix()
	print km_train
	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	print km_test

if __name__=='__main__':
	from tools.load import LoadMatrix
	lm=LoadMatrix()
	fm_train_dna=lm.load_dna('../data/fm_train_dna.dat')
	fm_test_dna=lm.load_dna('../data/fm_test_dna.dat')
	kernel_linear_string_modular(*parameter_list[0])
