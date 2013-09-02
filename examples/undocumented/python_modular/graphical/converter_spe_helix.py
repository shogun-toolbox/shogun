"""
Shogun demo

Fernando J. Iglesias Garcia

This example shows the use of dimensionality reduction methods, mainly 
Stochastic Proximity Embedding (SPE), although Isomap is also used for 
comparison. The data selected to be embedded is an helix. Two different methods
of SPE (global and local) are applied showing that the global method outperforms
the local one in this case. Actually the results of local SPE are fairly poor 
for this input. Finally, the reduction achieved with Isomap is better than the 
two previous ones, more robust against noise. Isomap exploits the 
parametrization of the input data.
"""

import math
import mpl_toolkits.mplot3d as mpl3
import numpy as np
import pylab
import util

from modshogun  import RealFeatures
from modshogun import StochasticProximityEmbedding, SPE_GLOBAL
from modshogun import SPE_LOCAL, Isomap

# Number of data points
N = 500

# Generate helix
t = np.linspace(1, N, N).T / N 
t = t*2*math.pi
X = np.r_[ [ ( 2 + np.cos(8*t) ) * np.cos(t) ],
           [ ( 2 + np.cos(8*t) ) * np.sin(t) ],
           [ np.sin(8*t) ] ]

# Bi-color helix
labels = np.round( (t*1.5) ) % 2

y1 = labels == 1
y2 = labels == 0

# Plot helix

fig = pylab.figure()

fig.add_subplot(2, 2, 1, projection = '3d')

pylab.plot(X[0, y1], X[1, y1], X[2, y1], 'ro')
pylab.plot(X[0, y2], X[1, y2], X[2, y2], 'go')

pylab.title('Original 3D Helix')

# Create features instance
features = RealFeatures(X)

# Create Stochastic Proximity Embedding converter instance
converter = StochasticProximityEmbedding()

# Set target dimensionality
converter.set_target_dim(2)
# Set strategy
converter.set_strategy(SPE_GLOBAL)

# Compute SPE embedding
embedding = converter.embed(features)

X = embedding.get_feature_matrix()

fig.add_subplot(2, 2, 2)

pylab.plot(X[0, y1], X[1, y1], 'ro')
pylab.plot(X[0, y2], X[1, y2], 'go')

pylab.title('SPE with global strategy')

# Compute a second SPE embedding with local strategy
converter.set_strategy(SPE_LOCAL)
converter.set_k(12)
embedding = converter.embed(features)

X = embedding.get_feature_matrix()

fig.add_subplot(2, 2, 3)

pylab.plot(X[0, y1], X[1, y1], 'ro')
pylab.plot(X[0, y2], X[1, y2], 'go')

pylab.title('SPE with local strategy')

# Compute Isomap embedding (for comparison)
converter = Isomap()
converter.set_target_dim(2)
converter.set_k(6)

embedding = converter.embed(features)

X = embedding.get_feature_matrix()

fig.add_subplot(2, 2, 4)

pylab.plot(X[0, y1], X[1, y1], 'ro')
pylab.plot(X[0, y2], X[1, y2], 'go')

pylab.title('Isomap')

pylab.connect('key_press_event', util.quit)
pylab.show()
