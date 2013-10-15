#!/usr/bin/env python

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# Written (W) 2011 Sergey Lisitsyn
# Copyright (C) 2011 Sergey Lisitsyn

from modshogun import *
from numpy import *
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
import re,os,time
from pylab import *

def build_features(path):
	files = os.listdir(path)
	files.remove('README')
	N = len(files)
	(nd,md) = imread(os.path.join(path,files[0])).shape
	dim = nd*md
	feature_matrix = zeros([dim,N])
	for i,filename in enumerate(files):
		feature_matrix[:,i] = imread(os.path.join(path,filename)).ravel()
	return nd,md,RealFeatures(feature_matrix)

path = '../../data/faces/'
converter = DiffusionMaps
nd,md,features = build_features(path)
converter_instance = converter()
converter_instance.set_t(5)
converter_instance.set_target_dim(2)

start = time.time()
new_features = converter_instance.embed(features).get_feature_matrix()
print new_features.shape
end = time.time()

clusterer = KMeans
clusterer_instance = clusterer(2,EuclideanDistance())
clusterer_instance.train(features)
labels = clusterer_instance.apply().get_labels()
print labels

print 'applied %s, took %fs' % (converter_instance.get_name(), end-start)
print 'plotting'

fig = figure()
ax = fig.add_subplot(111,axisbg='#ffffff')
ax.scatter(new_features[0],new_features[1],color='black')
import random
for i in range(len(new_features[0])):
	feature_vector = features.get_feature_vector(i)
	Z = zeros([nd,md,4])
	Z[:,:,0] = 255-feature_vector.reshape(nd,md)[::-1,:]
	Z[:,:,1] = Z[:,:,0]
	Z[:,:,2] = Z[:,:,0]
	for k in range(nd):
		for j in range(md):
			Z[k,j,3] = pow(sin(k*pi/nd)*sin(j*pi/md),0.5)
	imagebox = OffsetImage(Z,cmap=cm.gray,zoom=0.25)
	ab = AnnotationBbox(imagebox, (new_features[0,i],new_features[1,i]),
						pad=0.001,frameon=False)
	ax.add_artist(ab)
axis('off')
savefig('faces.png')
show()
