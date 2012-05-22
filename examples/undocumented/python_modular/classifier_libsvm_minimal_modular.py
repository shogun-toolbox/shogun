from numpy import *
from numpy.random import randn
from shogun.Features import *
from shogun.Classifier import *
from shogun.Kernel import *

num=1000
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

labels=BinaryLabels(trainlab);
svm=LibSVM(C, kernel, labels);
svm.train();

kernel.init(feats_train, feats_test);
out=svm.apply().get_labels();
testerr=mean(sign(out)!=testlab)
print(testerr)
