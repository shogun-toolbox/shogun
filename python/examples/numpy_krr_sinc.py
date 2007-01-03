from pylab import figure,pcolor,scatter,contour,colorbar,show,subplot,plot
from numpy import array,meshgrid,reshape,linspace,ones,min,max
from numpy import concatenate,transpose,ravel,double,sinc,pi
from numpy.random import randn, rand
from shogun.Features import *
from shogun.Classifier import *
from shogun.Kernel import *

X = 4*rand(1, 100) - 2; X.sort()
Y = sinc(pi*X) + 0.1*randn(1, 100)

width=1

feat = RealFeatures(X)
lab = Labels(Y.flatten())
print lab
gk=GaussianKernel(feat,feat, width)
krr = KRR()
krr.set_labels(lab)
krr.set_kernel(gk)
krr.set_tau(1e-6)
krr.train()

plot(X, Y, '.')
plot(X, krr.classify().get_labels(), hold=True)

# compute output plot iso-lines
XE = 4*rand(1, 500) - 2; XE.sort();
feat_test=RealFeatures(XE)
gk.init(feat, feat_test, True)
YE = krr.classify().get_labels()

plot(XE, YE, hold=True)
#plot(XE[0,200], array([krr.classify_example(200)]), '+', hold=True)
YE200 = krr.classify_example(200)
print YE[200], YE200
show()
