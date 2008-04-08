#!/usr/bin/env python
"""
Explicit examples on how to use clustering
"""

from numpy import array, concatenate, sign
from numpy.random import rand, seed, permutation
from sg import sg

def get_clouds (num, num_feats, num_vec):
	data=[rand(num_feats, num_vec)+x/2 for x in xrange(num)]
	cloud=concatenate(data, axis=1)
	return array([permutation(x) for x in cloud])

def kmeans ():
	print 'KMeans'

	size_cache=10
	k=3
	iter=1000
	num_feats=14
	num_trainvec=10

	trainlab=sign(rand(1, num_trainvec*k)-0.5)[0]
	traindata=get_clouds(k, num_feats, num_trainvec)

	sg('set_features', 'TRAIN', traindata)
	sg('set_labels', 'TRAIN', trainlab)
	sg('send_command', 'set_distance EUCLIDIAN REAL')
	sg('send_command', 'init_distance TRAIN')
	sg('send_command', 'new_classifier KMEANS')
	sg('send_command', 'train_classifier %d %d' % (k, iter))

	[radi, centers]=sg('get_classifier')


def hierarchical ():
	print 'Hierarchical'

	size_cache=10
	merges=3
	num_feats=14
	num_trainvec=10

	traindata=get_clouds(merges, num_feats, num_trainvec)

	sg('set_features', 'TRAIN', traindata)
	sg('send_command', 'set_distance EUCLIDIAN REAL')
	sg('send_command', 'init_distance TRAIN')
	sg('send_command', 'new_classifier HIERARCHICAL')
	sg('send_command', 'train_classifier %d' % merges)

	[merge_distance, pairs]=sg('get_classifier')

###########################################################################
# call functions
###########################################################################

if __name__=='__main__':
	kmeans()
	hierarchical()
