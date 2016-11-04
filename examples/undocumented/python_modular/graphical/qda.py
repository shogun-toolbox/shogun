from modshogun import RealFeatures
from modshogun import MulticlassLabels
from modshogun import QDA
from pylab import pcolor, contour, colorbar, connect, show, plot, axis

import numpy as np
import util

N = 500
size = 100


# positive examples
mean_pos = [-1, 4]
cov_pos  = [[1,40], [50, -2]]

x_pos, y_pos = np.random.multivariate_normal(mean_pos, cov_pos, 500).T
plot(x_pos, y_pos, 'bo');

# negative examples
mean_neg = [0, -3]
cov_neg  = [[100,50], [20, 3]]

x_neg, y_neg = np.random.multivariate_normal(mean_neg, cov_neg, 500).T
plot(x_neg, y_neg, 'ro');

# train qda
labels = MulticlassLabels( np.concatenate([np.zeros(N), np.ones(N)]) )
pos = np.array([x_pos, y_pos])
neg = np.array([x_neg, y_neg])
features = RealFeatures( np.array(np.concatenate([pos, neg], 1)) )

qda = QDA()
qda.set_labels(labels)
qda.train(features)

# compute output plot iso-lines
xs = np.array(np.concatenate([x_pos, x_neg]))
ys = np.array(np.concatenate([y_pos, y_neg]))

x1_max = max(1.2*xs)
x1_min = min(1.2*xs)
x2_max = max(1.2*ys)
x2_min = min(1.2*ys)

x1 = np.linspace(x1_min, x1_max, size)
x2 = np.linspace(x2_min, x2_max, size)

x, y = np.meshgrid(x1, x2)

dense = RealFeatures( np.array((np.ravel(x), np.ravel(y))) )
dense_labels = qda.apply(dense).get_labels()

z = dense_labels.reshape((size, size))

pcolor(x, y, z)
contour(x, y, z, linewidths = 1, colors = 'black', hold = True)

axis([x1_min, x1_max, x2_min, x2_max])

connect('key_press_event', util.quit)

show()
