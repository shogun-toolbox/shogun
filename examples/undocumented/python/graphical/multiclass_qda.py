"""
Shogun demo

Fernando J. Iglesias Garcia
"""

import numpy as np
import matplotlib as mpl
import pylab
import util

from scipy import linalg
from modshogun import QDA
from modshogun import RealFeatures, MulticlassLabels

# colormap
cmap = mpl.colors.LinearSegmentedColormap('color_classes',
	{'red':   [(0, 1,  1),
		   (1, .7, .7)],
	 'green': [(0, 1, 1),
	           (1, .7, .7)],
	 'blue':  [(0, 1, 1),
	           (1, .7, .7)]})
pylab.cm.register_cmap(cmap = cmap)

# Generate data from Gaussian distributions
def gen_data():
	np.random.seed(0)
	covs = np.array([[[0., -1. ], [2.5,  .7]],
			 [[3., -1.5], [1.2, .3]],
			 [[ 2,  0  ], [ .0,  1.5 ]]])
	X = np.r_[np.dot(np.random.randn(N, dim), covs[0]) + np.array([-4, 3]),
		  np.dot(np.random.randn(N, dim), covs[1]) + np.array([-1, -5]),
		  np.dot(np.random.randn(N, dim), covs[2]) + np.array([3, 4])];
	Y = np.hstack((np.zeros(N), np.ones(N), 2*np.ones(N)))
	return X, Y

def plot_data(qda, X, y, y_pred, ax):
	X0, X1, X2 = X[y == 0], X[y == 1], X[y == 2]

	# Correctly classified
	tp = (y == y_pred)
	tp0, tp1, tp2 = tp[y == 0], tp[y == 1], tp[y == 2]
	X0_tp, X1_tp, X2_tp = X0[tp0], X1[tp1], X2[tp2]

	# Misclassified
	X0_fp, X1_fp, X2_fp = X0[tp0 != True], X1[tp1 != True], X2[tp2 != True]

	# Class 0 data
	pylab.plot(X0_tp[:, 0], X0_tp[:, 1], 'o', color = cols[0])
	pylab.plot(X0_fp[:, 0], X0_fp[:, 1], 's', color = cols[0])
	m0 = qda.get_mean(0)
	pylab.plot(m0[0], m0[1], 'o', color = 'black', markersize = 8)

	# Class 1 data
	pylab.plot(X1_tp[:, 0], X1_tp[:, 1], 'o', color = cols[1])
	pylab.plot(X1_fp[:, 0], X1_fp[:, 1], 's', color = cols[1])
	m1 = qda.get_mean(1)
	pylab.plot(m1[0], m1[1], 'o', color = 'black', markersize = 8)

	# Class 2 data
	pylab.plot(X2_tp[:, 0], X2_tp[:, 1], 'o', color = cols[2])
	pylab.plot(X2_fp[:, 0], X2_fp[:, 1], 's', color = cols[2])
	m2 = qda.get_mean(2)
	pylab.plot(m2[0], m2[1], 'o', color = 'black', markersize = 8)

def plot_cov(plot, mean, cov, color):
	v, w = linalg.eigh(cov)
	u = w[0] / linalg.norm(w[0])
	angle = np.arctan(u[1] / u[0])	# rad
	angle = 180 * angle / np.pi	# degrees
	# Filled gaussian at 2 standard deviation
	ell = mpl.patches.Ellipse(mean, 2*v[0]**0.5, 2*v[1]**0.5, 180 + angle, color = color)
	ell.set_clip_box(plot.bbox)
	ell.set_alpha(0.5)
	plot.add_artist(ell)

def plot_regions(qda):
	nx, ny = 500, 500
	x_min, x_max = pylab.xlim()
	y_min, y_max = pylab.ylim()
	xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx),
			     np.linspace(y_min, y_max, ny))
	dense = RealFeatures(np.array((np.ravel(xx), np.ravel(yy))))
	dense_labels = qda.apply(dense).get_labels()
	Z = dense_labels.reshape(xx.shape)
	pylab.pcolormesh(xx, yy, Z)
	pylab.contour(xx, yy, Z, linewidths = 3, colors = 'k')

# Number of classes
M = 3
# Number of samples of each class
N = 300
# Dimension of the data
dim = 2

cols = ['blue', 'green', 'red']

fig = pylab.figure()
ax  = fig.add_subplot(111)
pylab.title('Quadratic Discrimant Analysis')

X, y = gen_data()

labels = MulticlassLabels(y)
features = RealFeatures(X.T)
qda = QDA(features, labels, 1e-4, True)
qda.train()
ypred = qda.apply().get_labels()

plot_data(qda, X, y, ypred, ax)
for i in range(M):
	plot_cov(ax, qda.get_mean(i), qda.get_cov(i), cols[i])
plot_regions(qda)

pylab.connect('key_press_event', util.quit)
pylab.show()
