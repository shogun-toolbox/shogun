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

def pbm2numpy(filename):
	fin = None
	try:
		fin = open(filename, 'rb')
		fin.readline()
		#fin.readline()
		header = fin.readline().strip()
		match = re.match('^(\d+) (\d+)$', header)
		colsr, rowsr = match.groups()
		rows,cols = int(rowsr),int(colsr)
		result = zeros((rows, cols))
		fin.readline()
		for i in range(rows):
			for j in range(cols):
				char1 = fin.read(1)
				#fin.read(1)
				result[i, j] = ord(char1)
		return (result.ravel(), rows,cols)
	finally:
		if fin != None:
			fin.close()
		fin = None
	return None

def build_features(path):
	files = os.listdir(path)
	files.remove('README')
	N = len(files)
	first,nd,md = pbm2numpy(os.path.join(path,files[0]))
	dim = nd*md
	feature_matrix = zeros([dim,N])
	for i,filename in enumerate(files):
		feature_matrix[:,i], q,p = pbm2numpy(os.path.join(path,filename))
	return nd,md,RealFeatures(feature_matrix)

path = '../../data/faces/'
converter = DiffusionMaps
nd,md,features = build_features(path)
converter_instance = converter()
converter_instance.parallel.set_num_threads(2)
converter_instance.set_t(1)
#converter_instance.set_kernel(GaussianKernel(50,500000.0))
#converter_instance.set_nullspace_shift(1e-3)
converter_instance.set_target_dim(2)

start = time.time()
new_features = converter_instance.embed(features).get_feature_matrix()
end = time.time()

clusterer = KMeans
clusterer_instance = clusterer(2,EuclidianDistance())
clusterer_instance.train(features)
labels = clusterer_instance.apply().get_labels()
print labels

print 'applied %s, took %fs' % (converter_instance.get_name(), end-start)
print 'plotting'

fig = figure()
ax = fig.add_subplot(111,axisbg='#ffffff')
ax.scatter(new_features[0],new_features[1],color='black')
for i in range(len(new_features[0])):
	feature_vector = features.get_feature_vector(i)
	Z = zeros([nd,md,4])
	Z[:,:,0] = 255-feature_vector.reshape(nd,md)
	Z[:,:,1] = Z[:,:,0]
	Z[:,:,2] = Z[:,:,0]-30*(labels[i])
	Z[Z[:,:,2]<0] = 0
	for k in range(nd):
		for j in range(md):
			Z[k,j,3] = pow(sin(k*pi/nd)*sin(j*pi/md),0.5)
	imagebox = OffsetImage(Z,cmap=cm.gray,zoom=0.60)
	ab = AnnotationBbox(imagebox, (new_features[0,i],new_features[1,i]),
						pad=0.001,frameon=False)
	ax.add_artist(ab)
axis('off')
show()
