from pylab import figure,pcolor,scatter,contour,colorbar,show,subplot,plot,legend
from numpy import array,meshgrid,reshape,linspace,ones,min,max
from numpy import concatenate,transpose,ravel,double,sinc,pi
from numpy.random import randn, rand
from shogun.Features import *
from shogun.Regression import *
from shogun.Kernel import *

X = 4*rand(1, 100) - 2; X.sort()
Y = sinc(pi*X) + 0.1*randn(1, 100)

C=10
width=0.5
epsilon=0.01

feat = RealFeatures(X)
lab = Labels(Y.flatten())
gk=GaussianKernel(feat,feat, width)
#svr = SVRLight(C, epsilon, gk, lab)
svr = LibSVR(C, epsilon, gk, lab)
svr.train()

plot(X, Y, '.', label='train data')
plot(X[0], svr.classify().get_labels(), hold=True, label='train output')

# compute output plot iso-lines
XE = 4*rand(1, 500) - 2; XE.sort();
feat_test=RealFeatures(XE)
gk.init(feat, feat_test)
YE = svr.classify().get_labels()

plot(XE[0], YE, hold=True, label='test output')
show()
