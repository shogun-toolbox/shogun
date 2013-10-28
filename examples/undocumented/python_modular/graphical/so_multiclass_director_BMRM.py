#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

from modshogun	import RealFeatures
from modshogun	import MulticlassModel, MulticlassSOLabels, RealNumber, DualLibQPBMSOSVM, DirectorStructuredModel
from modshogun	import BMRM, PPBMRM, P3BMRM, ResultSet, RealVector
from modshogun	import StructuredAccuracy

class MulticlassStructuredModel(DirectorStructuredModel):
	def __init__(self,features,labels):
		DirectorStructuredModel.__init__(self)
		self.set_features(features)
		self.set_labels(labels)
		self.dim = features.get_dim_feature_space()*labels.get_num_classes()
		self.n_classes = labels.get_num_classes()
		self.n_feats = features.get_dim_feature_space()
		#self.use_director_risk()
	def get_dim(self):
		return self.dim
	def argmax(self,w,feat_idx,training):
		feature_vector = self.get_features().get_feature_vector(feat_idx)
		label = None
		if training == True:
			label = int(RealNumber.obtain_from_generic(self.get_labels().get_label(feat_idx)).value)
		ypred = 0
		max_score = -1e10
		for c in xrange(self.n_classes):
			score = 0.0
			for i in xrange(self.n_feats):
				score += w[i+self.n_feats*c]*feature_vector[i]
			if training == True:
				score += (c!=label)
			if score > max_score:
				max_score = score
				ypred = c

		res = ResultSet()
		res.score = max_score
		res.psi_pred = RealVector(self.dim)
		res.psi_pred.zero()
		for i in xrange(self.n_feats):
			res.psi_pred[i+self.n_feats*ypred] = feature_vector[i]
		res.argmax = RealNumber(ypred)
		if training == True:
			res.delta = (label!=ypred)
			res.psi_truth = RealVector(self.dim)
			res.psi_truth.zero()
			for i in xrange(self.n_feats):
				res.psi_truth[i+self.n_feats*label] = feature_vector[i]
			for i in xrange(self.n_feats):
				res.score -= w[i+self.n_feats*label]*feature_vector[i]
		return res


def fill_data(cnt, minv, maxv):
	x1 = np.linspace(minv, maxv, cnt)
	a, b = np.meshgrid(x1, x1)
	X = np.array((np.ravel(a), np.ravel(b)))
	y = np.zeros((1, cnt*cnt))
	tmp = cnt*cnt;
	y[0, tmp/3:(tmp/3)*2]=1
	y[0, tmp/3*2:(tmp/3)*3]=2
	return X, y.flatten()

def gen_data():
	covs = np.array([[[0., -1. ], [2.5,  .7]],
		[[3., -1.5], [1.2, .3]],
		[[ 2,  0  ], [ .0,  1.5 ]]])
	X = np.r_[np.dot(np.random.randn(N, dim), covs[0]) + np.array([0, 10]),
			np.dot(np.random.randn(N, dim), covs[1]) + np.array([-10, -10]),
			np.dot(np.random.randn(N, dim), covs[2]) + np.array([10, -10])];
	Y = np.hstack((np.zeros(N), np.ones(N), 2*np.ones(N)))
	return X, Y

def get_so_labels(out):
	N = out.get_num_labels()
	l = np.zeros(N)
	for i in xrange(N):
		l[i] = RealNumber.obtain_from_generic(out.get_label(i)).value
	return l

# Number of classes
M = 3
# Number of samples of each class
N = 10
# Dimension of the data
dim = 2

X, y = gen_data()

cnt = 50

X2, y2 = fill_data(cnt, np.min(X), np.max(X))

labels = MulticlassSOLabels(y)
features = RealFeatures(X.T)

model = MulticlassStructuredModel(features, labels)

lambda_ = 1e1
sosvm = DualLibQPBMSOSVM(model, labels, lambda_)

sosvm.set_cleanAfter(10)	# number of iterations that cutting plane has to be inactive for to be removed
sosvm.set_cleanICP(True)	# enables inactive cutting plane removal feature
sosvm.set_TolRel(0.001)		# set relative tolerance
sosvm.set_verbose(True)		# enables verbosity of the solver
sosvm.set_cp_models(16)		# set number of cutting plane models
sosvm.set_solver(BMRM)		# select training algorithm
#sosvm.set_solver(PPBMRM)
#sosvm.set_solver(P3BMRM)

sosvm.train()

res = sosvm.get_result()
Fps = res.get_hist_Fp_vector()
Fds = res.get_hist_Fd_vector()
wdists = res.get_hist_wdist_vector()

plt.figure()
plt.subplot(221)
plt.title('Fp and Fd history')
plt.plot(xrange(res.get_n_iters()), Fps, hold=True)
plt.plot(xrange(res.get_n_iters()), Fds, hold=True)
plt.subplot(222)
plt.title('w dist history')
plt.plot(xrange(res.get_n_iters()), wdists)

# Evaluation
out = sosvm.apply()

Evaluation = StructuredAccuracy()
acc = Evaluation.evaluate(out, labels)

print "Correct classification rate: %0.4f%%" % ( 100.0*acc )

# show figure
Z = get_so_labels(sosvm.apply(RealFeatures(X2)))
x = (X2[0,:]).reshape(cnt, cnt)
y = (X2[1,:]).reshape(cnt, cnt)
z = Z.reshape(cnt, cnt)

plt.subplot(223)
plt.pcolor(x, y, z, shading='interp')
plt.contour(x, y, z, linewidths=1, colors='black', hold=True)
plt.plot(X[:,0], X[:,1], 'yo')
plt.axis('tight')
plt.title('Classification')
plt.show()
