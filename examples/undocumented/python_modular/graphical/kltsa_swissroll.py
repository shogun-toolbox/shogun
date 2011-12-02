from modshogun import *
import numpy
import pylab
from time import time

N = 6000
numpy.random.seed(40)

tt = numpy.array((3*numpy.pi/2)*(1+2*numpy.random.rand(N)))
height = numpy.array((numpy.random.rand(N)-0.5))
noise = 0.0
X = numpy.array([(tt+noise*numpy.random.randn(N))*numpy.cos(tt), 10*height, (tt+noise*numpy.random.randn(N))*numpy.sin(tt)])

import matplotlib.pyplot as plt
import numpy
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
fig.set_facecolor('white')
ax = fig.add_subplot(121,projection='3d')
plt.title('3d swissroll data',fontsize=10)
cset = ax.scatter(X[0], X[1], X[2], s=10,c=tt, cmap=pylab.cm.Spectral)
ax.view_init(5, -70)
ax.dist = 7.5

for axis in ax.w_xaxis, ax.w_yaxis, ax.w_zaxis: 
  for elt in axis.get_ticklines() + axis.get_ticklabels(): 
    elt.set_visible(False) 
  axis.pane.set_visible(False) 
  axis.gridlines.set_visible(False) 
  axis.line.set_visible(False) 

features = RealFeatures(X)

converter = KernelLocalTangentSpaceAlignment()
#converter.set_nullspace_shift(0.0)
converter.set_target_dim(2)
converter.parallel.set_num_threads(1)
converter.set_k(25)

print 'Embedding..'
start = time()
new_feats = converter.embed(features).get_feature_matrix()
end = time()
print 'Time passed', end-start

print new_feats.shape
ax2 = fig.add_subplot(122)	
plot = ax2.scatter(new_feats[0],new_feats[1],c=tt, cmap=pylab.cm.Spectral)
ax2.axis('off')
plt.title('%s embedding' % converter.get_name(),fontsize=10)
plt.show()
