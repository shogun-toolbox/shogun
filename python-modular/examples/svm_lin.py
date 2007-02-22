from shogun.Classifier import SVMLin,M_DEBUG
from shogun.Features import SparseRealFeatures, RealFeatures,Labels
from numpy.random import randn
from numpy import array,double,concatenate,mean,sign,zeros

num=4000
distance=1
dims=5
asym=1
C=10
zerodims=5

traindat=concatenate( (randn(dims,num)-asym*distance, randn(dims,num)+distance), axis=1 )
traindat=concatenate( (zeros((zerodims,2*num)), traindat), axis=0)
trainlab=array(num*[-1]+num*[+1],dtype=double)

testdat=concatenate( (randn(dims,num)-asym*distance, randn(dims,num)+distance), axis=1 )
testdat=concatenate( (zeros((zerodims,2*num)), testdat), axis=0)
testlab=array(num*[-1]+num*[+1],dtype=double)

densefeat=RealFeatures(traindat)
feat=SparseRealFeatures()
feat.io.set_loglevel(M_DEBUG)
feat.obtain_from_simple(densefeat)
lab=Labels(trainlab)
l=SVMLin(C, feat, lab)
l.io.set_loglevel(M_DEBUG)
l.train()
b=l.get_bias()
w=l.get_w()

densefeat=RealFeatures(testdat)
feat=SparseRealFeatures()
feat.io.set_loglevel(M_DEBUG)
feat.obtain_from_simple(densefeat)
l.set_features(feat)
out=l.classify().get_labels()

print 'bias: %f' % b
print 'w: ' + `w`
print 'error: %f' % mean(sign(out)!=testlab)
