import numpy
tt = numpy.genfromtxt('../../../../../data/toy/hemisphere_color.dat',unpack=True).T
X = numpy.genfromtxt('../../../../../data/toy/hemisphere.dat',unpack=True).T
N = X.shape[1]
preprocs = []

from shogun.Preprocessor import LocallyLinearEmbedding
lle = LocallyLinearEmbedding()
lle.set_k(20)
preprocs.append((lle, "Locally Linear Embedding with k=%d" % lle.get_k()))

from shogun.Preprocessor import ClassicMDS
mds = ClassicMDS()
preprocs.append((mds, "Classic MDS"))

from shogun.Preprocessor import LandmarkMDS
lmds = LandmarkMDS()
lmds.set_landmark_number(50)
preprocs.append((lmds,"Landmark MDS with %d landmarks" % lmds.get_landmark_number()))

from shogun.Preprocessor import ClassicIsomap, KISOMAP
cisomap = ClassicIsomap()
cisomap.set_type(KISOMAP)
cisomap.set_k(9)
preprocs.append((cisomap,"Classic K-Isomap with k=%d" % cisomap.get_k()))

from shogun.Preprocessor import LandmarkIsomap
lisomap = LandmarkIsomap()
lisomap.set_landmark_number(50)
lisomap.set_type(KISOMAP)
lisomap.set_k(9)
preprocs.append((lisomap,"L-K-Isomap with k=%d, %d landmarks" % (lisomap.get_k(),lisomap.get_landmark_number())))

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
swiss_roll_fig = fig.add_subplot(len(preprocs)/2+1,len(preprocs)/2,1,projection='3d')
swiss_roll_fig.scatter(X[0], X[1], X[2], s=10, c=tt, cmap=plt.cm.Spectral)
plt.subplots_adjust(hspace=0.4)

from shogun.Features import RealFeatures

for (i, (preproc, label)) in enumerate(preprocs):
	X = numpy.genfromtxt('../../../../../data/toy/hemisphere.dat',unpack=True).T
	features = RealFeatures(X)
	preproc.set_target_dim(2)
	new_feats = preproc.apply_to_feature_matrix(features)
	preproc_subplot = fig.add_subplot(len(preprocs)/2+1,len(preprocs)/2,i+2)
	preproc_subplot.scatter(new_feats[0],new_feats[1], c=tt, cmap=plt.cm.Spectral)
	plt.title(label)
	print preproc.get_name(), 'done'
	
plt.show()
