"""
Blind Source Separation using the Jade Algorithm with Shogun

Based on the example from scikit-learn
http://scikit-learn.org/

Kevin Hughes 2013
"""

import matplotlib.pyplot as plt
import numpy as np
import shogun as sg

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
X = np.dot(A, S)
mixed_signals = sg.features(X)

# Separating
jade = sg.transformer('Jade')
jade.fit(mixed_signals)
signals = jade.transform(mixed_signals)
S_ = signals.get('feature_matrix')
A_ = jade.get('mixing_matrix')

# Plot results
plt.figure()
plt.subplot(3, 1, 1)
plt.plot(S.T)
plt.title('True Sources')
plt.subplot(3, 1, 2)
plt.plot(X.T)
plt.title('Mixed Sources')
plt.subplot(3, 1, 3)
plt.plot(S_.T)
plt.title('Estimated Sources')
plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.36)
plt.show()
