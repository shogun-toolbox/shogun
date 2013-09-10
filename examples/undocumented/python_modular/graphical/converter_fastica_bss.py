"""
Blind Source Separation using the FastICA Algorithm with Shogun

Based on the example from scikit-learn
http://scikit-learn.org/

Kevin Hughes 2013
"""


import numpy as np
import pylab as pl

from shogun.Features  import RealFeatures
from shogun.Converter import FastICA

# Generate sample data
np.random.seed(0)
n_samples = 2000
time = np.linspace(0, 10, n_samples)

# Source Signals
s1 = np.sin(2 * time)  # sin wave
s2 = np.sign(np.sin(3 * time))  # square wave
S = np.c_[s1, s2]
S += 0.2 * np.random.normal(size=S.shape)  # add noise

# Standardize data
S /= S.std(axis=0)  
S = S.T

# Mixing Matrix
A = np.array([[1, 0.5], [0.5, 1]])

# Mix Signals
X = np.dot(A,S)
mixed_signals = RealFeatures(X)

# Separating
ica = FastICA()
signals = ica.apply(mixed_signals)
S_ = signals.get_feature_matrix()
A_ = ica.get_mixing_matrix();

# Plot results
pl.figure()
pl.subplot(3, 1, 1)
pl.plot(S.T)
pl.title('True Sources')
pl.subplot(3, 1, 2)
pl.plot(X.T)
pl.title('Mixed Sources')
pl.subplot(3, 1, 3)
pl.plot(S_.T)
pl.title('Estimated Sources')
pl.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.36)
pl.show()
