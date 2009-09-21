def histogram ():
	print 'Histogram'

	order=3
	gap=0
	reverse='n' # bit silly to not use boolean, set 'r' to yield true
	from sg import sg

#	sg('new_distribution', 'HISTOGRAM')
	sg('add_preproc', 'SORTWORDSTRING')

	sg('set_features', 'TRAIN', fm_train, 'DNA')
	sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse)
	sg('attach_preproc', 'TRAIN')

#	sg('train_distribution')
#	histo=sg('get_histogram')

#	num_examples=11
#	num_param=sg('get_histogram_num_model_parameters')
#	for i in xrange(num_examples):
#		for j in xrange(num_param):
#			sg('get_log_derivative %d %d' % (j, i))

#	sg('get_log_likelihood')
#	sg('get_log_likelihood_sample')

if __name__=='__main__':
	from tools.load import LoadMatrix
	lm=LoadMatrix()
	fm_train=lm.load_dna('../data/fm_train_dna.dat')
	fm_cube=lm.load_cubes('../data/fm_train_cube.dat')
	histogram()
