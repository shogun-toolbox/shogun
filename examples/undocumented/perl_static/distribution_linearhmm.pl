from tools.load import LoadMatrix
from sg import sg
lm=LoadMatrix()
traindna=lm.load_dna('../data/fm_train_dna.dat')
cubedna=lm.load_cubes('../data/fm_train_cube.dat')
parameter_list=[[traindna,cubedna,3,0,'n'],
		[traindna,cubedna,3,0,'n']]

def distribution_linearhmm (fm_train=traindna,fm_cube=cubedna,
			   order=3,gap=0,reverse='n'):
#	sg('new_distribution', 'LinearHMM')
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

#	sg('get_log_likelihood_sample')


if __name__=='__main__':
	print('LinearHMM')
	distribution_linearhmm(*parameter_list[0])
