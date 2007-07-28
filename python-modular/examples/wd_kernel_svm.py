from numpy import *
from random import choice,seed
from shogun.Features import *
from shogun.Classifier import *
from shogun.Kernel import *

degree=20
num_dat=40
len=70
acgt=['A','C','G','T']
C=1
seed(17)

def gen_random_string():
	dat = num_dat*[[]]
	lab = concatenate((-ones(num_dat,dtype=double), ones(num_dat,dtype=double)))
	for i in range(num_dat):
		dat[i]=[choice(acgt) for j in range(len)]
		if lab[i]==1:
			dat[i][10:12]='A'
		dat[i]="".join(dat[i])
	sdat = StringCharFeatures(DNA)
	sdat.set_string_features(dat)
	slab = Labels(lab)
	return (sdat,slab)

#generate train and test data
(trainfeat,trainlab)=gen_random_string()
(testfeat,testlab)=gen_random_string()

#train svm
wdk = WeightedDegreeStringKernel(trainfeat,trainfeat, degree)
km=wdk.get_kernel_matrix()
print km
print trainlab.get_labels()
svm = LibSVM(C, wdk, trainlab)
svm.train()
print 'hi'
print svm.get_num_support_vectors()
trainout=svm.classify().get_labels()
print 'hi2'
svs=[ (svm.get_alpha(i),svm.get_support_vector(i)) for i in range(svm.get_num_support_vectors())]

#test
wdk.init(trainfeat,testfeat)
testout=svm.classify().get_labels()

print "\n classification error:" + `mean(sign(testout)!=telab)`
