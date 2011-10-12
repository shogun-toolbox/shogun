from modshogun import *
import numpy
numpy.random.seed(40)
N = 2000
tt = numpy.array((numpy.pi)*(3+2*numpy.random.rand(N)))
height = numpy.array(numpy.random.rand(N)-0.5)
X = numpy.array([tt*numpy.cos(tt), 10*height, tt*numpy.sin(tt)])
preprocs = []

lle = LocallyLinearEmbedding()
lle.set_k(9)
preprocs.append((lle, "LLE preset k"))

lle_adaptive_k = LocallyLinearEmbedding()
lle_adaptive_k.set_k(3)
lle_adaptive_k.set_max_k(20)
lle_adaptive_k.set_auto_k(True)
preprocs.append((lle_adaptive_k, "LLE auto k"))

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
new_mpl = False

try:
	swiss_roll_fig = fig.add_subplot(1,3,1, projection='3d')
	new_mpl = True
except:
	figure = plt.figure()
	swiss_roll_fig = Axes3D(figure)

swiss_roll_fig.scatter(X[0], X[1], X[2], s=10, c=tt, cmap=plt.cm.Spectral)

plt.subplots_adjust(wspace=0.3)
plt.title('3D data')
from shogun.Features import RealFeatures

for (i, (preproc, label)) in enumerate(preprocs):
	features = RealFeatures(X)
	preproc.set_target_dim(2)
	preproc.io.set_loglevel(MSG_DEBUG)
	new_feats = preproc.apply_to_feature_matrix(features)
	if not new_mpl:
		preproc_subplot = fig.add_subplot(1,3,i+1)
	else:
		preproc_subplot = fig.add_subplot(1,3,i+2)
	preproc_subplot.scatter(new_feats[0],new_feats[1], c=tt, cmap=plt.cm.Spectral)
	plt.axis('tight')
	plt.xticks([]), plt.yticks([])
	plt.title(label + ' (k=%d)' % preproc.get_k())
	
plt.show()
