from numpy import array,zeros,concatenate,uint8
from shogun.Features import *
from shogun.SVM import *
from shogun.Kernel import *

dat=zeros([10,100],dtype=uint8)
feat = ByteFeatures(dat)
lab = Labels(concatenate(ones(1,50),ones(1,50),axis=1))
wdk=WeightedDegreeCharKernel(feat,feat, 10,1)
svm = SVMLight(10, wdk, lab)
svm.train()
print svm.classify().get_labels()
print lab.get_labels()

import kernel.Kernel as k
import features.Features as f
import classifier.svm.SVM as svm
import numpy as N

dA = k.doubleArray(5)
dA[0] = 0.167
dA[1] = 0.25
dA[2] = 0.5
dA[3] = 0.25
dA[4] = 0.167

cA = N.chararray((12,1),1)
cA[0] = 'A'
cA[1] = 'A'
cA[2] = 'A'
cA[3] = 'A'
cA[4] = 'A'
cA[5] = 'A'
cA[6] = 'A'
cA[7] = 'A'
cA[8] = 'A'
cA[9] = 'A'
cA[10] = 'A'
cA[11] = 'A'

#f=cf.CCharFeatures(cf.DNA,'ACGTAAACCGGT',4,3)
feat123=f.CCharFeatures(f.DNA,4)
feat123.testNumpy(cA)
#print
#print 'test add param...'

#feat123.testNumpy2(N.zeros((5,10),'i'))
#feat123.testNumpy3(N.zeros((4,3),'d'))
#feat123.testNumpy3(N.array('[1.,2.,3.,4.,5.; 1.,2.,3.,4.,5.]'))
x=N.array([1.,2.,3.,4.,5.,6.])
x.shape=(2,3)
#feat123.testNumpy3(N.zeros((5,4),'c'))
#feat123.testNumpy3(x)
#print 'test add param2...'
feat123.testAddParam3(x,1)
#print 'test add param3...'
#print 'test add param...'
#feat123.testAddParam(1,cA,4,3)

#print
#print 'testing constructor...'
#feat123=f.CCharFeatures(f.DNA,cA,4,3)
#kernel = k.CWeightedDegreeCharKernel(feat123,feat123,10,dA,2,0,True,False,1)
#
#trainlabels = f.CLabels(3)
#trainlabels.set_int_label(0,1)
#trainlabels.set_int_label(1,-1)
#trainlabels.set_int_label(2,-1)
#
#s = svm.CSVMLight()
#s.set_labels(trainlabels)
#s.set_kernel(kernel)
#result = s.train();
#print result
