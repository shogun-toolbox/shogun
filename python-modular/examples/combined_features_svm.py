from numpy import *
from random import choice,seed
from numpy.random import randn
from shogun.Features import *
from shogun.Classifier import *
from shogun.Kernel import *

degree=20
num_dat=1000
dims=10
distance=1
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

	sdat1 = StringCharFeatures(DNA)
	sdat1.set_string_features(dat)

	dat = num_dat*[[]]
	for i in range(num_dat):
		dat[i]=[choice(acgt) for j in range(len)]
		if lab[i]==1:
			dat[i][20:22]=['A','A']
		dat[i]="".join(dat[i])

	sdat2 = StringCharFeatures(DNA)
	sdat2.set_string_features(dat)
	sdat3=RealFeatures(concatenate( (randn(dims,num_dat/2)-distance, randn(dims,num_dat/2)+distance), axis=1 ))

	slab = Labels(lab)
	return (sdat1,sdat2,sdat3, slab)

#generate train and test data
(trainfeat1, trainfeat2, trainfeat3, trainlab)=gen_random_string()
(testfeat1, testfeat2, testfeat3, testlab)=gen_random_string()

k=CombinedKernel()
k.append_kernel(CWeightedDegreeStringKernel(10, 0, degree, 0))
k.append_kernel(CWeightedDegreeStringKernel(10, 0, degree, 0))
k.append_kernel(GaussianKernel(10, 5.5))

f=CombinedFeatures()
f.append_feature_obj(trainfeat1)
f.append_feature_obj(trainfeat2)
f.append_feature_obj(trainfeat3)
k.init(f,f)

#train svm
svm = LibSVM(C, k, trainlab)
svm.train()
trainout=svm.classify().get_labels()
svs=[ (svm.get_alpha(i),svm.get_support_vector(i)) for i in range(svm.get_num_support_vectors())]

#test
f_test=CombinedFeatures()
f_test.append_feature_obj(testfeat1)
f_test.append_feature_obj(testfeat2)
f_test.append_feature_obj(testfeat3)
k.init(f,f_test)
testout=svm.classify().get_labels()

print "\n classification error:" + `mean(sign(testout)!=testlab.get_labels())`
