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
import matplotlib.pyplot as plt
import numpy as np
import shogun as sg
from mpl_toolkits import mplot3d

# Number of data points
N = 500

# Generate helix
t = np.linspace(1, N, N).T / N
t = t * 2 * math.pi
X = np.r_[[(2 + np.cos(8 * t)) * np.cos(t)],
          [(2 + np.cos(8 * t)) * np.sin(t)],
          [np.sin(8 * t)]]

# Bi-color helix
labels = np.round((t * 1.5)) % 2

y1 = labels == 1
y2 = labels == 0

# Plot helix

fig = plt.figure()

fig.add_subplot(2, 2, 1, projection='3d')

plt.plot(X[0, y1], X[1, y1], X[2, y1], 'ro')
plt.plot(X[0, y2], X[1, y2], X[2, y2], 'go')

plt.title('Original 3D Helix')

# Create features instance
features = sg.features(X)

# Create Stochastic Proximity Embedding converter instance
converter = sg.transformer('StochasticProximityEmbedding')

# Set target dimensionality
converter.put('target_dim', 2)
# Set strategy
converter.put('m_strategy', 'SPE_GLOBAL')

# Compute SPE embedding
embedding = converter.transform(features)

X = embedding.get('feature_matrix')

fig.add_subplot(2, 2, 2)

plt.plot(X[0, y1], X[1, y1], 'ro')
plt.plot(X[0, y2], X[1, y2], 'go')

plt.title('SPE with global strategy')

# Compute a second SPE embedding with local strategy
converter.put('m_strategy', 'SPE_LOCAL')
converter.put('m_k', 12)
embedding = converter.transform(features)

X = embedding.get('feature_matrix')

fig.add_subplot(2, 2, 3)

plt.plot(X[0, y1], X[1, y1], 'ro')
plt.plot(X[0, y2], X[1, y2], 'go')

plt.title('SPE with local strategy')

# Compute Isomap embedding (for comparison)
converter = sg.transformer('Isomap')
converter.put('target_dim', 2)
converter.put('k', 6)

embedding = converter.transform(features)

X = embedding.get('feature_matrix')

fig.add_subplot(2, 2, 4)

plt.plot(X[0, y1], X[1, y1], 'ro')
plt.plot(X[0, y2], X[1, y2], 'go')

plt.title('Isomap')

plt.show()
