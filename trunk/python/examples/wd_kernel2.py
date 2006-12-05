from numpy import *
from numpy.random import *
from shogun.Features import *
from shogun.Classifier import *
from shogun.Kernel import *

num_dat=50
len=70
acgt=array(['A','C','G','T'])

#generate train data
trdat=chararray((len,2*num_dat),1,order='FORTRAN')
trlab=concatenate((-ones(num_dat,dtype=double), ones(num_dat,dtype=double)))
for ix in xrange(2*num_dat):
    trdat[:,ix]=acgt[array(floor(4*random_sample(len)), dtype=int)]

trdat[10:15,trlab==1]='A'
        
trainfeat = CharFeatures(trdat,DNA)
trainlab = Labels(trlab)

#generate test data
tedat=chararray((len,2*num_dat),1,order='FORTRAN')
telab=concatenate((-ones(num_dat,dtype=double), ones(num_dat,dtype=double)))
for ix in xrange(2*num_dat):
    tedat[:,ix]=acgt[array(floor(4*random_sample(len)), dtype=int)]

tedat[10:15,telab==1]='A'
testfeat = CharFeatures(tedat,DNA)
testlab = Labels(telab)

#train svm
weights=ones(20,dtype=double)
for ix in xrange(20):
    weights[ix] = 1/(double(ix)+1);


wdk=WeightedDegreeCharKernel(trainfeat,trainfeat, 10, weights)
svm = SVMLight(10, wdk, trainlab)
svm.train()
print svm.get_num_support_vectors()

#trainout=svm.classify().get_labels()
K = mat(svm.get_kernel().get_kernel_matrix())
alphas = mat(zeros((2*num_dat,1)))
for ix in xrange(svm.get_num_support_vectors()):
    alphas[svm.get_support_vector(ix)] = svm.get_alpha(ix)

trainout = sign(K*alphas + svm.get_bias())
print trainout



#test
wdk_test=WeightedDegreeCharKernel(trainfeat,testfeat, 10, weights)
svm.set_kernel(wdk_test)
#svm.init_kernel_optimization()
#wdk.init(trainfeat,testfeat, True)
#testout=svm.classify().get_labels()
K = mat(svm.get_kernel().get_kernel_matrix())
testout = sign(K*alphas + svm.get_bias())
print testout



