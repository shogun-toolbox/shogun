from numpy import array
from numpy.random import rand
from shogun.Features import *
from shogun.SVM import *
from shogun.Kernel import *

feat = RealFeatures(rand(5,10))
lab = Labels(array([-1,1,1,-1,1,1,-1,-1,-1,1]))
gk=GaussianKernel(feat,feat, 10,1)
svm = SVMLight(10, gk, lab)
svm.train()
print svm.classify().get_labels()
print lab.get_labels()
