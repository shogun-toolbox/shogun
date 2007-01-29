from shogun.Classifier import LDA,M_DEBUG
from shogun.Features import RealFeatures,Labels
from numpy.random import randn
from numpy import array,double,concatenate,mean,sign

num=100000;
distance=1;
dims=10;
gamma=0;

traindat=concatenate( (randn(dims,num)-distance, randn(dims,num)+distance), axis=1 )
trainlab=array(num*[-1]+num*[+1],dtype=double)

testdat=concatenate( (randn(dims,num)-distance, randn(dims,num)+distance), axis=1 )
testlab=array(num*[-1]+num*[+1],dtype=double)

feat=RealFeatures(traindat)
lab=Labels(trainlab)
l=LDA(gamma, feat, lab)
l.train()
b=l.get_bias()
w=l.get_w()
feat=RealFeatures(testdat)
l.set_features(feat)
out=l.classify().get_labels()

print 'bias: %f' % b
print 'w: ' + `w`
print 'error: %f' % mean(sign(out)!=testlab)
