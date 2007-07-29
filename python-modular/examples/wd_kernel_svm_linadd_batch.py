from numpy import *
from random import choice,seed
from shogun.Features import *
from shogun.Classifier import *
from shogun.Kernel import *

C=100
degree=20
num_dat=100
seqlen=70
acgt=['A','C','G','T']
seed(17)

#create random data 
def gen_random_string():
	dat = num_dat*[[]]
	lab = concatenate((-ones(num_dat/2,dtype=double), ones(num_dat/2,dtype=double)))
	for i in range(num_dat):
		dat[i]=[choice(acgt) for j in range(seqlen)]
		if lab[i]==1:
			dat[i][10:12]=['A','C']
		dat[i]="".join(dat[i])
	sdat = StringCharFeatures(DNA)
	sdat.set_string_features(dat)
	slab = Labels(lab)
	return (dat,sdat,slab)

#generate train and test data
(trainstrs,trainfeat,trainlab)=gen_random_string()
(teststrs,testfeat,testlab)=gen_random_string()

#train svm
weights = arange(1,degree+1,dtype=double)[::-1]/sum(arange(1,degree+1,dtype=double))

wdk=WeightedDegreeStringKernel(trainfeat,trainfeat, degree, weights=weights)
svm = LibSVM(C, wdk, trainlab)
svm.set_epsilon(1e-8)
svm.train()
print 'LibSVM Objective: %f num_sv: %d' % (svm.get_objective(), svm.get_num_support_vectors())
svm.set_batch_computation_enabled(False)
svm.set_linadd_enabled(False)
trainout=svm.classify().get_labels()
svm.set_batch_computation_enabled(True)
btrainout=svm.classify().get_labels()
K0 = mat(wdk.get_kernel_matrix())

assert (max(abs(btrainout-trainout).flat) < 1e-6)

alphas0 = zeros(num_dat)
for ix in xrange(svm.get_num_support_vectors()):
    alphas0[svm.get_support_vector(ix)] = svm.get_alpha(ix)
trainout0 = K0*alphas0.flat + svm.get_bias()

assert (max(abs(btrainout-trainout0).flat) < 1e-6)

alphas1 = zeros(num_dat)
for ix in xrange(svm.get_num_support_vectors()):
    alphas1[svm.get_support_vector(ix)] = svm.get_alpha(ix)

al=svm.get_alphas()
balphas =  zeros(num_dat)
balphas[svm.get_support_vectors()]=svm.get_alphas()
assert(max(abs(alphas0-balphas)) < 1e-16)

wdk.init(trainfeat,testfeat)
testout0=svm.classify().get_labels()
K1 = wdk.get_kernel_matrix()
wdk_test=WeightedDegreeStringKernel(trainfeat,testfeat, degree)
K2 = wdk_test.get_kernel_matrix()
svm.set_kernel(wdk_test)
testout1=svm.classify().get_labels()
testout2=mat(K1).T*balphas.flat + svm.get_bias()

b4=svm.get_bias()
alphas4=svm.get_alphas()
svs4=svm.get_support_vectors()
ntrdat=[ trainstrs[i] for i in svs4 ]
svs4=arange(len(alphas4))
dat=StringCharFeatures(DNA)
dat.set_string_features(ntrdat)
wdk_test4=WeightedDegreeStringKernel(dat, testfeat,degree)
svm4=SVM(wdk_test4, alphas4, svs4, b4)
print "out3"
testout3=svm4.classify().get_labels()
svm.set_batch_computation_enabled(False)
print "out4"
testout4=svm4.classify().get_labels()
K4=svm4.get_kernel().get_kernel_matrix()
testout5=mat(K4).T*alphas4.flat + svm.get_bias()
print svm4.get_kernel().get_is_initialized()
print svm4.get_num_support_vectors()


svm6 = SVMLight(C, wdk, trainlab)
svm6.set_epsilon(1e-8)
wdk.init(trainfeat,trainfeat)
svm6.train()
print 'SVMLight Objective: %f num_sv: %d' % (svm6.get_objective(), svm6.get_num_support_vectors())
wdk.init(trainfeat,testfeat)
testout6=array([ svm6.classify_example(i) for i in xrange(testfeat.get_num_vectors()) ])

print '0-1:' + `max(abs(testout0-testout1).flat)`
print '0-2:' + `max(abs(testout0-testout2).flat)`
print '0-3:' + `max(abs(testout0-testout3).flat)`
print '0-4:' + `max(abs(testout0-testout4).flat)`
print '0-5:' + `max(abs(testout0-testout5).flat)`
print '0-6:' + `max(abs(testout0-testout6).flat)`
print '1-2:' + `max(abs(testout1-testout2).flat)`
print '1-3:' + `max(abs(testout1-testout3).flat)`
print '1-4:' + `max(abs(testout1-testout4).flat)`
print '2-3:' + `max(abs(testout2-testout3).flat)`
print '2-4:' + `max(abs(testout2-testout4).flat)`
print '3-4:' + `max(abs(testout1-testout4).flat)`
