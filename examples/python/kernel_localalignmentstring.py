def local_alignment_string():
	print 'LocalAlignmentString'

	size_cache=10

	from sg import sg
	sg('set_features', 'TRAIN', fm_train_dna, 'DNA')
	sg('set_features', 'TEST', fm_test_dna, 'DNA')
	sg('set_kernel', 'LOCALALIGNMENT', 'CHAR', size_cache)
	sg('init_kernel', 'TRAIN')
	km=sg('get_kernel_matrix')

	sg('init_kernel', 'TEST')
	km=sg('get_kernel_matrix')

if __name__=='__main__':
	from tools.load import LoadMatrix
	lm=LoadMatrix()
	fm_train_dna=lm.load_dna('../data/fm_train_dna.dat')
	fm_test_dna=lm.load_dna('../data/fm_test_dna.dat')
	local_alignment_string()
