from numpy import *
from numpy.random import *
from shogun.Features import *
from shogun.Classifier import *
from shogun.Kernel import *

degree=20
num_dat=500
len=70
acgt=array(['A','C','G','T'])
C=1

seed(17)
#generate train data
trdat = chararray((len,2*num_dat),1,order='FORTRAN')
trlab = concatenate((-ones(num_dat,dtype=double), ones(num_dat,dtype=double)))
for i in range(len):
	trdat[i,:]=acgt[array(floor(4*random_sample(2*num_dat)), dtype=int)]

trdat[10:12,trlab==1]='A'
trainfeat = CharFeatures(trdat,DNA)
trainlab = Labels(trlab)

#generate test data
tedat = chararray((len,2*num_dat),1,order='FORTRAN')
telab = concatenate((-ones(num_dat,dtype=double), ones(num_dat,dtype=double)))
for i in range(len):
	tedat[i,:]=acgt[array(floor(4*random_sample(2*num_dat)), dtype=int)]

tedat[10:12,telab==1]='A'
testfeat = CharFeatures(tedat,DNA)
testlab = Labels(telab)

#train svm
wdk = WeightedDegreeCharKernel(trainfeat,trainfeat, degree)
svm = SVMLight(C, wdk, trainlab)
svm.train()
print svm.get_num_support_vectors()
trainout=svm.classify().get_labels()
svs=[ (svm.get_alpha(i),svm.get_support_vector(i)) for i in range(svm.get_num_support_vectors())]

#test
wdk.init(trainfeat,testfeat)
testout=svm.classify().get_labels()

print "\n classification error:" + `mean(sign(testout)!=telab)`
