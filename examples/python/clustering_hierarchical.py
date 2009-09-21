def hierarchical ():
	print 'Hierarchical'

	size_cache=10
	merges=3

	from sg import sg
	sg('set_features', 'TRAIN', fm_train)
	sg('set_distance', 'EUCLIDIAN', 'REAL')
	sg('init_distance', 'TRAIN')
	sg('new_clustering', 'HIERARCHICAL')
	sg('train_clustering', merges)

	[merge_distance, pairs]=sg('get_clustering')

if __name__=='__main__':
	from tools.load import LoadMatrix
	lm=LoadMatrix()
	fm_train=lm.load_numbers('../data/fm_train_real.dat')
	hierarchical()
