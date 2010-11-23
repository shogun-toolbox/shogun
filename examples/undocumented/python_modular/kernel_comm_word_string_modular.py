from tools.load import LoadMatrix
lm=LoadMatrix()

parameter_list = [[lm.load_dna('../data/fm_train_dna.dat'),lm.load_dna('../data/fm_test_dna.dat'),4,0,False, False],[lm.load_dna('../data/fm_train_dna.dat'),lm.load_dna('../data/fm_test_dna.dat'),4,0,False,False]]

def kernel_comm_word_string_modular (fm_train_dna=lm.load_dna('../data/fm_train_dna.dat'), fm_test_dna=lm.load_dna('../data/fm_test_dna.dat'), order=3, gap=0, reverse = False, use_sign = False):
	print 'CommWordString'
	from shogun.Kernel import CommWordStringKernel
	from shogun.Features import StringWordFeatures, StringCharFeatures, DNA
	from shogun.PreProc import SortWordString
	fm_train_dna     = fm_train_dna
	fm_test_dna      = fm_test_dna
	order            = order
	gap              = gap
	reverse          = reverse
	use_sign         = use_sign


	charfeat=StringCharFeatures(DNA)
	charfeat.set_features(fm_train_dna)
	feats_train=StringWordFeatures(charfeat.get_alphabet())
	feats_train.obtain_from_char(charfeat, order-1, order, gap, reverse)
	preproc=SortWordString()
	preproc.init(feats_train)
	feats_train.add_preproc(preproc)
	feats_train.apply_preproc()

	charfeat=StringCharFeatures(DNA)
	charfeat.set_features(fm_test_dna)
	feats_test=StringWordFeatures(charfeat.get_alphabet())
	feats_test.obtain_from_char(charfeat, order-1, order, gap, reverse)
	feats_test.add_preproc(preproc)
	feats_test.apply_preproc()



	kernel=CommWordStringKernel(feats_train, feats_train, use_sign)

	km_train=kernel.get_kernel_matrix()
	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	print km_test

if __name__=='__main__':
	from tools.load import LoadMatrix
	lm=LoadMatrix()
	fm_train_dna=lm.load_dna('../data/fm_train_dna.dat')
	fm_test_dna=lm.load_dna('../data/fm_test_dna.dat')
	kernel_comm_word_string_modular(*parameter_list[0])
