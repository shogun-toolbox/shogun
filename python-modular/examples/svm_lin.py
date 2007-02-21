from shogun.Classifier import SVMLin,M_DEBUG
from shogun.Features import SparseRealFeatures, RealFeatures,Labels
from numpy.random import randn
from numpy import array,double,concatenate,mean,sign

num=100
distance=1
dims=10
C=1

traindat=concatenate( (randn(dims,num)-distance, randn(dims,num)+distance), axis=1 )
trainlab=array(num*[-1]+num*[+1],dtype=double)

testdat=concatenate( (randn(dims,num)-distance, randn(dims,num)+distance), axis=1 )
testlab=array(num*[-1]+num*[+1],dtype=double)

densefeat=RealFeatures(traindat)
feat=SparseRealFeatures()
feat.obtain_from_simple(densefeat)
lab=Labels(trainlab)
l=SVMLin(C, feat, lab)
l.io.set_loglevel(M_DEBUG)
l.train()
#b=l.get_bias()
#w=l.get_w()
#feat=RealFeatures(testdat)
#l.set_features(feat)
#out=l.classify().get_labels()

#print 'bias: %f' % b
#print 'w: ' + `w`
#print 'error: %f' % mean(sign(out)!=testlab)
