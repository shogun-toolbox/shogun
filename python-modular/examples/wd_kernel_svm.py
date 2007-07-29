from numpy import *
from random import choice,seed
from shogun.Features import *
from shogun.Classifier import *
from shogun.Kernel import *

degree=20
num_dat=1000
len=70
acgt=['A','C','G','T']
C=1
seed(17)

#create random data 
def gen_random_string():
	dat = num_dat*[[]]
	lab = concatenate((-ones(num_dat/2,dtype=double), ones(num_dat/2,dtype=double)))
	for i in range(num_dat):
		dat[i]=[choice(acgt) for j in range(len)]
		if lab[i]==1:
			dat[i][10:12]=['A','A']
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
svm = LibSVM(C, wdk, trainlab)
svm.train()
trainout=svm.classify().get_labels()
svs=[ (svm.get_alpha(i),svm.get_support_vector(i)) for i in range(svm.get_num_support_vectors())]

#test
wdk.init(trainfeat,testfeat)
testout=svm.classify().get_labels()

print "\n classification error:" + `mean(sign(testout)!=testlab.get_labels())`
