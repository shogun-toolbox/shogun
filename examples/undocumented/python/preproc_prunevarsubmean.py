def prune_var_sub_mean ():
	print 'PruneVarSubMean'

	width=1.4
	size_cache=10
	divide_by_std=True

	from sg import sg
	sg('add_preproc', 'PRUNEVARSUBMEAN', divide_by_std)
	sg('set_kernel', 'CHI2', 'REAL', size_cache, width)

	sg('set_features', 'TRAIN', fm_train_real)
	sg('attach_preproc', 'TRAIN')
	km=sg('get_kernel_matrix', 'TRAIN')

	sg('set_features', 'TEST', fm_test_real)
	sg('attach_preproc', 'TEST')
	km=sg('get_kernel_matrix', 'TEST')

if __name__=='__main__':
	from tools.load import LoadMatrix
	lm=LoadMatrix()
	fm_train_real=lm.load_numbers('../data/fm_train_real.dat')
	fm_test_real=lm.load_numbers('../data/fm_test_real.dat')
	prune_var_sub_mean()
