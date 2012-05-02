#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt

from shogun.Features import Labels, RealFeatures
from shogun.Regression import LeastAngleRegression, LinearRidgeRegression, LeastSquaresRegression

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
yall = Xall[:,0] + Xall[:,1] + Xall[:,2] + 2*np.random.randn(n)

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
LeastAngleRegression.set_labels(Labels(y))
LeastAngleRegression.train(RealFeatures(X.T))

# train ordinary LSR
if use_ridge:
    lsr = LinearRidgeRegression(0.01, RealFeatures(X.T), Labels(y))
    lsr.train()
else:
    lsr = LeastSquaresRegression()
    lsr.set_labels(Labels(y))
    lsr.train(RealFeatures(X.T))

# gather LASSO path
path = np.zeros((p, LeastAngleRegression.get_path_size()))
for i in xrange(path.shape[1]):
    path[:,i] = LeastAngleRegression.get_w(i)

# apply on training data
mse_train = np.zeros(LeastAngleRegression.get_path_size())
for i in xrange(mse_train.shape[0]):
    LeastAngleRegression.switch_w(i)
    ypred = LeastAngleRegression.apply(RealFeatures(X.T)).get_labels()
    mse_train[i] = np.dot(ypred - y, ypred - y) / y.shape[0]
ypred = lsr.apply(RealFeatures(X.T)).get_labels()
mse_train_lsr = np.dot(ypred - y, ypred - y) / y.shape[0]

# apply on test data
mse_test = np.zeros(LeastAngleRegression.get_path_size())
for i in xrange(mse_test.shape[0]):
    LeastAngleRegression.switch_w(i)
    ypred = LeastAngleRegression.apply(RealFeatures(Xtest.T)).get_labels()
    mse_test[i] = np.dot(ypred - ytest, ypred - ytest) / ytest.shape[0]
ypred = lsr.apply(RealFeatures(Xtest.T)).get_labels()
mse_test_lsr = np.dot(ypred - ytest, ypred - ytest) / ytest.shape[0]

fig = plt.figure()
ax_path = fig.add_subplot(1,2,1)
plt.plot(xrange(path.shape[1]), path.T, '.-')
plt.legend(['%d' % (x+1) for x in xrange(path.shape[0])])
plt.xlabel('iteration')
plt.title('LASSO path')

ax_tr   = fig.add_subplot(2,2,2)
plt.plot(xrange(mse_train.shape[0]), mse_train, 'k.-')
plt.plot(xrange(mse_train.shape[0]), np.zeros(mse_train.shape[0]) + mse_train_lsr, 'r-')
plt.legend(('LASSO', 'LeastSquares'))
plt.xlabel('# of non-zero variables')
plt.ylabel('MSE')
plt.title('MSE on training data')

ax_tt   = fig.add_subplot(2,2,4)
plt.plot(xrange(mse_test.shape[0]), mse_test, 'k.-')
plt.plot(xrange(mse_test.shape[0]), np.zeros(mse_test.shape[0]) + mse_test_lsr, 'r-')
plt.legend(('LASSO', 'LeastSquares'), loc='lower right')
plt.xlabel('# of non-zero variables')
plt.ylabel('MSE')
plt.title('MSE on test data')

plt.show()

