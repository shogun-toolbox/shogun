from numpy import array,double
from numpy.random import rand
from shogun.Features import *
from shogun.Classifier import *
from shogun.Kernel import *

feat = RealFeatures(rand(5,10))
lab = Labels(array([-1,1,1,-1,1,1,-1,-1,-1,1],dtype=double))
gk=GaussianKernel(feat,feat, 1)
krr = KRR()
krr.set_labels(lab)
krr.set_kernel(gk)
krr.set_tau(1e-5)
krr.train()
print krr.classify().get_labels()
print lab.get_labels()

feat_test = RealFeatures(rand(5,10))
gk.init(feat,feat_test,True)
output_test = krr.classify().get_labels() 
print output_test
