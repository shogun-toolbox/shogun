def plugin_estimate_histogram ():
	print 'PluginEstimate w/ HistogramWord'

	size_cache=10
	order=3
	gap=0
	reverse='n' # bit silly to not use boolean, set 'r' to yield true
	use_sign=False
	normalization='FULL'

	from sg import sg
	sg('set_features', 'TRAIN', fm_train_dna, 'DNA')
	sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse)

	sg('set_features', 'TEST', fm_test_dna, 'DNA')
	sg('convert', 'TEST', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse)

	pseudo_pos=1e-1
	pseudo_neg=1e-1
	sg('new_plugin_estimator', pseudo_pos, pseudo_neg)
	sg('set_labels', 'TRAIN', label_train_dna)
	sg('train_estimator')

	sg('set_kernel', 'HISTOGRAM', 'WORD', size_cache)
	sg('init_kernel', 'TRAIN')
	km=sg('get_kernel_matrix')

	sg('init_kernel', 'TEST')
# not supported yet
#	lab=sg('plugin_estimate_classify')
	km=sg('get_kernel_matrix')

if __name__=='__main__':
	from tools.load import LoadMatrix
	lm=LoadMatrix()
	fm_train_dna=lm.load_dna('../data/fm_train_dna.dat')
	fm_test_dna=lm.load_dna('../data/fm_test_dna.dat')
	label_train_dna=lm.load_labels('../data/label_train_dna.dat')
	plugin_estimate_histogram()
