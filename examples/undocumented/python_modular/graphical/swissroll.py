import numpy
numpy.random.seed(40)
tt = numpy.genfromtxt('../../../../data/toy/swissroll_color.dat',unpack=True).T
X = numpy.genfromtxt('../../../../data/toy/swissroll.dat',unpack=True).T
N = X.shape[1]
preprocs = []

from shogun.Preprocessor import LocallyLinearEmbedding
lle = LocallyLinearEmbedding()
lle.set_k(9)
preprocs.append((lle, "LLE with k=%d" % lle.get_k()))

from shogun.Preprocessor import MultidimensionalScaling
mds = MultidimensionalScaling()
preprocs.append((mds, "Classic MDS"))

lmds = MultidimensionalScaling()
lmds.set_landmark(True)
lmds.set_landmark_number(20)
preprocs.append((lmds,"Landmark MDS with %d landmarks" % lmds.get_landmark_number()))

from shogun.Preprocessor import Isomap
cisomap = Isomap()
cisomap.set_k(9)
preprocs.append((cisomap,"Isomap with k=%d" % cisomap.get_k()))

lisomap = Isomap()
lisomap.set_landmark(True)
lisomap.set_landmark_number(20)
lisomap.set_k(9)
preprocs.append((lisomap,"Landmark Isomap with k=%d, %d landmarks" % (lisomap.get_k(),lisomap.get_landmark_number())))

from shogun.Preprocessor import HessianLocallyLinearEmbedding
hlle = HessianLocallyLinearEmbedding()
hlle.set_k(6)
preprocs.append((hlle,"Hessian LLE with k=%d" % (hlle.get_k())))

from shogun.Preprocessor import LocalTangentSpaceAlignment
ltsa = LocalTangentSpaceAlignment()
ltsa.set_k(6)
preprocs.append((ltsa,"LTSA with k=%d" % (ltsa.get_k())))

from shogun.Preprocessor import LaplacianEigenmaps
le = LaplacianEigenmaps()
le.set_k(15)
le.set_tau(25.0)
preprocs.append((le,"Laplacian Eigenmaps with k=%d, tau=%d" % (le.get_k(),le.get_tau())))

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()

if matplotlib.__version__[0]=='0':
	figure = plt.figure()
	swiss_roll_fig = Axes3D(figure)
else:
	swiss_roll_fig = fig.add_subplot(3,3,1, projection='3d')

swiss_roll_fig.scatter(X[0], X[1], X[2], s=10, c=tt, cmap=plt.cm.Spectral)
plt.subplots_adjust(hspace=0.4)

from shogun.Features import RealFeatures

for (i, (preproc, label)) in enumerate(preprocs):
	X = numpy.genfromtxt('../../../../data/toy/swissroll.dat',unpack=True).T
	features = RealFeatures(X)
	preproc.set_target_dim(2)
	new_feats = preproc.apply_to_feature_matrix(features)
	if matplotlib.__version__[0]=='0':
		preproc_subplot = fig.add_subplot(4,2,i+1)
	else:
		preproc_subplot = fig.add_subplot(3,3,i+2)
	preproc_subplot.scatter(new_feats[0],new_feats[1], c=tt, cmap=plt.cm.Spectral)
	plt.title(label)
	print preproc.get_name(), 'done'
	
plt.show()
