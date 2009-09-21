def poly_match_word ():
	print 'PolyMatchWord'

	size_cache=10
	degree=2
	inhomogene=True
	normalize=True
	order=3
	gap=0
	reverse='n' # bit silly to not use boolean, set 'r' to yield true

	from sg import sg
	sg('add_preproc', 'SORTWORDSTRING')
	sg('set_features', 'TRAIN', fm_train_dna, 'DNA')
	sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse)
	sg('attach_preproc', 'TRAIN')

	sg('set_features', 'TEST', fm_test_dna, 'DNA')
	sg('convert', 'TEST', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse)
	sg('attach_preproc', 'TEST')

	sg('set_kernel', 'POLYMATCH', 'WORD', size_cache, degree, inhomogene, normalize)
	sg('init_kernel', 'TRAIN')
	km=sg('get_kernel_matrix')

	sg('init_kernel', 'TEST')
	km=sg('get_kernel_matrix')

if __name__=='__main__':
	from tools.load import LoadMatrix
	lm=LoadMatrix()
	fm_train_dna=lm.load_dna('../data/fm_train_dna.dat')
	fm_test_dna=lm.load_dna('../data/fm_test_dna.dat')
	label_train_dna=lm.load_labels('../data/label_train_dna.dat')
	poly_match_word()
