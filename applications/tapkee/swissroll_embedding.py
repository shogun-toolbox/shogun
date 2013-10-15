import numpy
numpy.random.seed(40)
tt = numpy.genfromtxt('../../data/toy/swissroll_color.dat',unpack=True).T
X = numpy.genfromtxt('../../data/toy/swissroll.dat',unpack=True).T
N = X.shape[1]
converters = []

from shogun.Converter import LocallyLinearEmbedding
lle = LocallyLinearEmbedding()
lle.set_k(9)
converters.append((lle, "LLE with k=%d" % lle.get_k()))

from shogun.Converter import MultidimensionalScaling
mds = MultidimensionalScaling()
converters.append((mds, "Classic MDS"))

lmds = MultidimensionalScaling()
lmds.set_landmark(True)
lmds.set_landmark_number(20)
converters.append((lmds,"Landmark MDS with %d landmarks" % lmds.get_landmark_number()))

from shogun.Converter import Isomap
cisomap = Isomap()
cisomap.set_k(9)
converters.append((cisomap,"Isomap with k=%d" % cisomap.get_k()))

from shogun.Converter import DiffusionMaps
from shogun.Kernel import GaussianKernel
dm = DiffusionMaps()
dm.set_t(2)
dm.set_width(1000.0)
converters.append((dm,"Diffusion Maps with t=%d, sigma=%.1f" % (dm.get_t(),dm.get_width())))

from shogun.Converter import HessianLocallyLinearEmbedding
hlle = HessianLocallyLinearEmbedding()
hlle.set_k(6)
converters.append((hlle,"Hessian LLE with k=%d" % (hlle.get_k())))

from shogun.Converter import LocalTangentSpaceAlignment
ltsa = LocalTangentSpaceAlignment()
ltsa.set_k(6)
converters.append((ltsa,"LTSA with k=%d" % (ltsa.get_k())))

from shogun.Converter import LaplacianEigenmaps
le = LaplacianEigenmaps()
le.set_k(20)
le.set_tau(100.0)
converters.append((le,"Laplacian Eigenmaps with k=%d, tau=%d" % (le.get_k(),le.get_tau())))

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()

new_mpl = False

try:
	swiss_roll_fig = fig.add_subplot(3,3,1, projection='3d')
	new_mpl = True
except:
	figure = plt.figure()
	swiss_roll_fig = Axes3D(figure)

swiss_roll_fig.scatter(X[0], X[1], X[2], s=10, c=tt, cmap=plt.cm.Spectral)
swiss_roll_fig._axis3don = False
plt.suptitle('Swissroll embedding',fontsize=9)
plt.subplots_adjust(hspace=0.4)

from shogun.Features import RealFeatures

for (i, (converter, label)) in enumerate(converters):
	X = numpy.genfromtxt('../../data/toy/swissroll.dat',unpack=True).T
	features = RealFeatures(X)
	converter.set_target_dim(2)
	converter.parallel.set_num_threads(1)
	new_feats = converter.embed(features).get_feature_matrix()
	if not new_mpl:
		embedding_subplot = fig.add_subplot(4,2,i+1)
	else:
		embedding_subplot = fig.add_subplot(3,3,i+2)
	embedding_subplot.scatter(new_feats[0],new_feats[1], c=tt, cmap=plt.cm.Spectral)
	plt.axis('tight')
	plt.xticks([]), plt.yticks([])
	plt.title(label,fontsize=9)
	print converter.get_name(), 'done'

plt.show()
