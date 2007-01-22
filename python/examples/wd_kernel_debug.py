from numpy import *
from numpy.random import *
from shogun.Features import *
from shogun.Classifier import *
from shogun.Kernel import *

num_dat=50
len=70
acgt=array(['A','C','G','T'])

seed(17)
#generate train data
trdat=chararray((len,2*num_dat),1,order='FORTRAN')
trlab=concatenate((-ones(num_dat,dtype=double), ones(num_dat,dtype=double)))
for i in range(len):
	trdat[i,:]=acgt[array(floor(4*random_sample(2*num_dat)), dtype=int)]
trdat[10:15,trlab==1]='A'
trainfeat = CharFeatures(trdat,DNA)
trainlab = Labels(trlab)

#generate test data
tedat=chararray((len,2*num_dat),1,order='FORTRAN')
telab=concatenate((-ones(num_dat,dtype=double), ones(num_dat,dtype=double)))
for i in range(len):
	tedat[i,:]=acgt[array(floor(4*random_sample(2*num_dat)), dtype=int)]
tedat[10:15,telab==1]='A'
testfeat = CharFeatures(tedat,DNA)
testlab = Labels(telab)

#train svm
weights2=array([1,2,3,4,5],dtype=double)/15
wdk2=WeightedDegreeCharKernel(trainfeat,trainfeat, 10, weights2)
k2=wdk2.get_kernel_matrix()

weights=array([5,4,3,3,1],dtype=double)/15
wdk=WeightedDegreeCharKernel(trainfeat,trainfeat, 10, weights)
k=wdk.get_kernel_matrix()

print k-k2
