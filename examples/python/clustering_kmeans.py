def kmeans ():
	print 'KMeans'

	size_cache=10
	k=3
	iter=1000

	from sg import sg
	sg('set_features', 'TRAIN', fm_train)
	sg('set_distance', 'EUCLIDIAN', 'REAL')
	sg('init_distance', 'TRAIN')
	sg('new_clustering', 'KMEANS')
	sg('train_clustering', k, iter)

	[radi, centers]=sg('get_clustering')

if __name__=='__main__':
	from tools.load import LoadMatrix
	lm=LoadMatrix()
	fm_train=lm.load_numbers('../data/fm_train_real.dat')
	kmeans()
