from numpy import array,double
from numpy.random import rand
from shogun.Features import *
from shogun.Classifier import *
from shogun.Kernel import *

feat = RealFeatures(rand(5,10))
lab = Labels(array([-1,1,1,-1,1,1,-1,-1,-1,1],dtype=double))
gk=GaussianKernel(feat,feat, 1)
svm = SVMLight(10, gk, lab)
svm.train()
print svm.classify().get_labels()
print lab.get_labels()

feat_test = RealFeatures(rand(5,10))
gk.init(feat,feat_test)
output_test = svm.classify().get_labels() 
