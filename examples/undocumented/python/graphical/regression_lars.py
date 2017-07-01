#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt

from modshogun import RegressionLabels, RealFeatures
from modshogun import LeastAngleRegression, LinearRidgeRegression, LeastSquaresRegression
from modshogun import MeanSquaredError

# we compare LASSO with ordinary least-squares (OLE)
# in the ideal case, the MSE of OLE should coincide
# with LASSO at the end of the path
#
# if OLE is unstable, we may use RidgeRegression (with
# a small regularization coefficient) to simulate OLE
use_ridge = False

np.random.seed(1024)

n           = 200
ntrain      = 100
p           = 7
correlation = 0.6

mean = np.zeros(p)
cov  = correlation*np.ones((p,p)) + (1-correlation)*np.eye(p)

Xall = np.random.multivariate_normal(mean, cov, n)

# model is the linear combination of the first three variables plus noise
yall = 2*Xall[:,0] + 5*Xall[:,1] + -3*Xall[:,2] + 0.5*np.random.randn(n)

X = Xall[0:ntrain,:]
y = yall[0:ntrain]

Xtest = Xall[ntrain:,:]
ytest = yall[ntrain:]

# preprocess data
for i in xrange(p):
    X[:,i] -= np.mean(X[:,i])
    X[:,i] /= np.linalg.norm(X[:,i])
y -= np.mean(y)

# train LASSO
LeastAngleRegression = LeastAngleRegression()
LeastAngleRegression.set_labels(RegressionLabels(y))
LeastAngleRegression.train(RealFeatures(X.T))

# train ordinary LSR
if use_ridge:
    lsr = LinearRidgeRegression(0.01, RealFeatures(X.T), Labels(y))
    lsr.train()
else:
    lsr = LeastSquaresRegression()
    lsr.set_labels(RegressionLabels(y))
    lsr.train(RealFeatures(X.T))

# gather LASSO path
path = np.zeros((p, LeastAngleRegression.get_path_size()))
for i in xrange(path.shape[1]):
    path[:,i] = LeastAngleRegression.get_w(i)

evaluator = MeanSquaredError()

# apply on training data
mse_train = np.zeros(LeastAngleRegression.get_path_size())
for i in xrange(mse_train.shape[0]):
    LeastAngleRegression.switch_w(i)
    ypred = LeastAngleRegression.apply(RealFeatures(X.T))
    mse_train[i] = evaluator.evaluate(ypred, RegressionLabels(y))
ypred = lsr.apply(RealFeatures(X.T))
mse_train_lsr = evaluator.evaluate(ypred, RegressionLabels(y))

# apply on test data
mse_test = np.zeros(LeastAngleRegression.get_path_size())
for i in xrange(mse_test.shape[0]):
    LeastAngleRegression.switch_w(i)
    ypred = LeastAngleRegression.apply(RealFeatures(Xtest.T))
    mse_test[i] = evaluator.evaluate(ypred, RegressionLabels(y))
ypred = lsr.apply(RealFeatures(Xtest.T))
mse_test_lsr = evaluator.evaluate(ypred, RegressionLabels(y))

fig = plt.figure()
ax_path = fig.add_subplot(1,2,1)
plt.plot(xrange(path.shape[1]), path.T, '.-')
plt.legend(['%d' % (x+1) for x in xrange(path.shape[0])])
plt.xlabel('iteration')
plt.title('LASSO path')

ax_tr   = fig.add_subplot(2,2,2)
plt.plot(range(mse_train.shape[0])[1:], mse_train[1:], 'k.-')
plt.plot(range(mse_train.shape[0])[1:], np.zeros(mse_train.shape[0])[1:] + mse_train_lsr, 'r-')
plt.legend(('LASSO', 'LeastSquares'))
plt.xlabel('# of non-zero variables')
plt.ylabel('MSE')
plt.title('MSE on training data')

ax_tt   = fig.add_subplot(2,2,4)
plt.plot(range(mse_test.shape[0])[1:], mse_test[1:], 'k.-')
plt.plot(range(mse_test.shape[0])[1:], np.zeros(mse_test.shape[0])[1:] + mse_test_lsr, 'r-')
plt.legend(('LASSO', 'LeastSquares'), loc='lower right')
plt.xlabel('# of non-zero variables')
plt.ylabel('MSE')
plt.title('MSE on test data')

plt.show()

