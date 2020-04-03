"""
Blind Source Separation using the UWedgeSep Algorithm with Shogun

Kevin Hughes 2013
"""

import matplotlib.pyplot as plt
import numpy as np
import shogun as sg

# Generate sample data
np.random.seed(0)
n_samples = 2000
time = np.linspace(0, 1, n_samples)

# Source Signals
s1 = np.sin(2 * 3.14 * 55 * time)  # sin wave
s2 = np.cos(2 * 3.14 * 100 * time)  # cos wave
S = np.c_[s1, s2]
S += 0.1 * np.random.normal(size=S.shape)  # add noise

# Standardize data
S /= S.std(axis=0)
S = S.T

# Mixing Matrix
A = np.array([[1, 0.85], [0.55, 1]])
# print A

# Mix Signals
X = np.dot(A, S)
mixed_signals = sg.features(X)

# Separating
uwedge = sg.transformer('UWedgeSep')
uwedge.fit(mixed_signals)
signals = uwedge.transform(mixed_signals)
S_ = signals.get('feature_matrix')
A_ = uwedge.get('mixing_matrix')
# print A_

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
