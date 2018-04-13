#!/usr/bin/env python

# This software is distributed under BSD 3-clause license (see LICENSE file).
#
# Authors: Sergey Lisitsyn

from numpy import *
from pylab import *
from shogun import *
import random
import difflib

def word_kernel(words):
	N = len(words)
	dist_matrix = zeros([N,N])
	for i in range(N):
		for j in range(i,N):
			s = difflib.SequenceMatcher(None,words[i],words[j])
			dist_matrix[i,j] = s.ratio()
	dist_matrix = 0.5*(dist_matrix+dist_matrix.T)
	return CustomKernel(dist_matrix)

print 'loading'
words = []
f = open("../../data/toy/words.dat")
for line in f:
	words.append(line[:-1])
f.close()
print 'loaded'

converter = KernelLocallyLinearEmbedding()
converter.set_k(10)
converter.set_target_dim(2)
converter.parallel.set_num_threads(1)
embedding = converter.embed_kernel(word_kernel(words[:200]))
embedding_matrix = embedding.get_feature_matrix()
fig = figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(embedding_matrix[0,:],embedding_matrix[1,:],alpha=0.4,cmap=cm.Spectral,c=embedding_matrix[0,:]*embedding_matrix[1,:])

# hardcode ;)
words_to_show = ['finishing','publishing','standing',\
                 'shifted','insisted','tilted','blasted',\
                 'jumble','battle','gobble']

for i in xrange(0,200):
	if words[i] in words_to_show:
		ax.text(embedding_matrix[0,i]*1.1,1.25*embedding_matrix[1,i],words[i],fontsize=16,alpha=1.0)

axis('off')
show()

