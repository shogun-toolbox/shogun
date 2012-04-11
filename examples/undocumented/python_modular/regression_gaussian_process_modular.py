###########################################################################
# Mean prediction from Gaussian Processes based on classifier_libsvm_minimal_modular.py
###########################################################################
from numpy import *
from numpy.random import randn
from shogun.Features import *
from shogun.Classifier import *
from shogun.Kernel import *

num=100
dist=1
width=2.1
C=1

traindata_real=concatenate((randn(2,num)-dist, randn(2,num)+dist), axis=1)
testdata_real=concatenate((randn(2,num)-dist, randn(2,num)+dist), axis=1);

trainlab=concatenate((-ones(num), ones(num)));
testlab=concatenate((-ones(num), ones(num)));

feats_train=RealFeatures(traindata_real);
feats_test=RealFeatures(testdata_real);
kernel=GaussianKernel(feats_train, feats_train, width);

labels=Labels(trainlab);
gp=GaussianProcessRegression(1.0, kernel, feats_train, labels);
gp.train(feats_train);
out=gp.apply(feats_test).get_labels();
testerr=mean(sign(out)!=testlab)
print testerr