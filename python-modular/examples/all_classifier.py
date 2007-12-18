#!/usr/bin/env python
"""
Explicit examples on how to use the different classifiers
"""

from numpy import double, array, floor, concatenate
from numpy.random import rand, seed, permutation
from shogun.Kernel import GaussianKernel, WeightedDegreeStringKernel
from shogun.Distance import EuclidianDistance
from shogun.Features import *
from shogun.Classifier import *

def get_clouds (num, num_feats, num_vec):
	data=[rand(num_feats, num_vec)+x/2 for x in xrange(num)]
	cloud=concatenate(data, axis=1)
	return array([permutation(x) for x in cloud])

def get_dna ():
	acgt=array(['A', 'C', 'G','T'])
	len_acgt=len(acgt)
	rand_train=[]
	rand_test=[]

	for i in xrange(11):
		str1=[]
		str2=[]
		for j in range(60):
			str1.append(acgt[floor(len_acgt*rand())])
			str2.append(acgt[floor(len_acgt*rand())])
		rand_train.append(''.join(str1))
	rand_test.append(''.join(str2))
	
	for i in xrange(6):
		str1=[]
		for j in range(60):
			str1.append(acgt[floor(len_acgt*rand())])
	rand_test.append(''.join(str1))

	return {'train': rand_train, 'test': rand_test}

###########################################################################
# kernel-based SVMs
###########################################################################

def svm_light ():
	print 'SVMLight'

	data=get_dna()
	feats_train=StringCharFeatures(DNA)
	feats_train.set_string_features(data['train'])
	feats_test=StringCharFeatures(DNA)
	feats_test.set_string_features(data['test'])
	degree=20

	kernel=WeightedDegreeStringKernel(feats_train, feats_train, degree)

	C=0.017
	epsilon=1e-5
	tube_epsilon=1e-2
	num_threads=1
	lab=rand(feats_train.get_num_vectors()).round()*2-1
	labels=Labels(array(lab))

	svm=SVMLight(C, kernel, labels)
	svm.set_epsilon(epsilon)
	svm.set_tube_epsilon(tube_epsilon)
	svm.parallel.set_num_threads(num_threads)
	svm.train()

	kernel.init(feats_train, feats_test)
	svm.classify().get_labels()

def libsvm ():
	print 'LibSVM'

	num_feats=9
	data=get_clouds(2, num_feats, 11)
	feats_train=RealFeatures(data)
	data=get_clouds(2, num_feats, 17)
	feats_test=RealFeatures(data)
	width=2.1
	kernel=GaussianKernel(feats_train, feats_train, width)

	C=0.017
	epsilon=1e-5
	tube_epsilon=1e-2
	num_threads=2
	lab=rand(feats_train.get_num_vectors()).round()*2-1
	labels=Labels(array(lab))

	svm=LibSVM(C, kernel, labels)
	svm.set_epsilon(epsilon)
	svm.set_tube_epsilon(tube_epsilon)
	svm.parallel.set_num_threads(num_threads)
	svm.train()

	kernel.init(feats_train, feats_test)
	svm.classify().get_labels()

def gpbtsvm ():
	print 'GPBTSVM'

	num_feats=9
	data=get_clouds(2, num_feats, 11)
	feats_train=RealFeatures(data)
	data=get_clouds(2, num_feats, 17)
	feats_test=RealFeatures(data)
	width=2.1
	kernel=GaussianKernel(feats_train, feats_train, width)

	C=0.017
	epsilon=1e-5
	tube_epsilon=1e-2
	num_threads=8
	lab=rand(feats_train.get_num_vectors()).round()*2-1
	labels=Labels(array(lab))

	svm=GPBTSVM(C, kernel, labels)
	svm.set_epsilon(epsilon)
	svm.set_tube_epsilon(tube_epsilon)
	svm.parallel.set_num_threads(num_threads)
	svm.train()

	kernel.init(feats_train, feats_test)
	svm.classify().get_labels()

def mpdsvm ():
	print 'MPDSVM'

	num_feats=9
	data=get_clouds(2, num_feats, 11)
	feats_train=RealFeatures(data)
	data=get_clouds(2, num_feats, 17)
	feats_test=RealFeatures(data)
	width=2.1
	kernel=GaussianKernel(feats_train, feats_train, width)

	C=0.017
	epsilon=1e-5
	tube_epsilon=1e-2
	num_threads=1
	lab=rand(feats_train.get_num_vectors()).round()*2-1
	labels=Labels(array(lab))

	svm=MPDSVM(C, kernel, labels)
	svm.set_epsilon(epsilon)
	svm.set_tube_epsilon(tube_epsilon)
	svm.parallel.set_num_threads(num_threads)
	svm.train()

	kernel.init(feats_train, feats_test)
	svm.classify().get_labels()

def libsvm_multiclass ():
	print 'LibSVMMultiClass'

	num_feats=9
	data=get_clouds(5, num_feats, 11)
	feats_train=RealFeatures(data)
	data=get_clouds(5, num_feats, 17)
	feats_test=RealFeatures(data)
	width=2.1
	kernel=GaussianKernel(feats_train, feats_train, width)

	C=0.017
	epsilon=1e-5
	tube_epsilon=1e-2
	num_threads=8
	lab=[double(x) for x in xrange(feats_train.get_num_vectors())]
	labels=Labels(array(lab))

	svm=LibSVMMultiClass(C, kernel, labels)
	svm.set_epsilon(epsilon)
	svm.set_tube_epsilon(tube_epsilon)
	svm.parallel.set_num_threads(num_threads)
	svm.train()

	kernel.init(feats_train, feats_test)
	svm.classify().get_labels()

def libsvm_oneclass ():
	print 'LibSVMOneClass'

	num_feats=9
	data=get_clouds(2, num_feats, 11)
	feats_train=RealFeatures(data)
	data=get_clouds(2, num_feats, 17)
	feats_test=RealFeatures(data)
	width=2.1
	kernel=GaussianKernel(feats_train, feats_train, width)

	C=0.017
	epsilon=1e-5
	tube_epsilon=1e-2
	num_threads=1

	svm=LibSVMOneClass(C, kernel)
	svm.set_epsilon(epsilon)
	svm.set_tube_epsilon(tube_epsilon)
	svm.parallel.set_num_threads(num_threads)
	svm.train()

	kernel.init(feats_train, feats_test)
	svm.classify().get_labels()

def gmnpsvm ():
	print 'GMNPSVM'

	num_feats=9
	data=get_clouds(4, num_feats, 11)
	feats_train=RealFeatures(data)
	data=get_clouds(4, num_feats, 17)
	feats_test=RealFeatures(data)
	width=2.1
	kernel=GaussianKernel(feats_train, feats_train, width)

	C=0.017
	epsilon=1e-5
	tube_epsilon=1e-2
	num_threads=1
	lab=[double(x) for x in xrange(feats_train.get_num_vectors())]
	labels=Labels(array(lab))

	svm=GMNPSVM(C, kernel, labels)
	svm.set_epsilon(epsilon)
	svm.set_tube_epsilon(tube_epsilon)
	svm.parallel.set_num_threads(num_threads)
	svm.train()

	kernel.init(feats_train, feats_test)
	svm.classify().get_labels()

###########################################################################
# run with batch or linadd on LibSVM
###########################################################################

def do_batch_linadd ():
	print 'LibSVM batch'

	data=get_dna()
	feats_train=StringCharFeatures(DNA)
	feats_train.set_string_features(data['train'])
	feats_test=StringCharFeatures(DNA)
	feats_test.set_string_features(data['test'])
	degree=20

	kernel=WeightedDegreeStringKernel(feats_train, feats_train, degree)

	C=0.017
	epsilon=1e-5
	tube_epsilon=1e-2
	num_threads=2
	lab=rand(feats_train.get_num_vectors()).round()*2-1
	labels=Labels(array(lab))

	svm=LibSVM(C, kernel, labels)
	svm.set_epsilon(epsilon)
	svm.set_tube_epsilon(tube_epsilon)
	svm.parallel.set_num_threads(num_threads)
	svm.train()

	kernel.init(feats_train, feats_test)

	print 'LibSVM Objective: %f num_sv: %d' % \
		(svm.get_objective(), svm.get_num_support_vectors())
	svm.set_batch_computation_enabled(False)
	svm.set_linadd_enabled(False)
	svm.classify().get_labels()

	svm.set_batch_computation_enabled(True)
	svm.classify().get_labels()

###########################################################################
# linear SVMs
###########################################################################

def subgradient_svm ():
	print 'SubGradientSVM'

	num_feats=11
	data=get_clouds(2, num_feats, 11)
	realfeat=RealFeatures(data)
	feats_train=SparseRealFeatures()
	feats_train.obtain_from_simple(realfeat)
	data=get_clouds(2, num_feats, 21)
	realfeat=RealFeatures(data)
	feats_test=SparseRealFeatures()
	feats_test.obtain_from_simple(realfeat)

	C=0.42
	epsilon=1e-3
	num_threads=1
	max_train_time=1.
	lab=rand(feats_train.get_num_vectors()).round()*2-1
	labels=Labels(array(lab))

	svm=SubGradientSVM(C, feats_train, labels)
	svm.set_epsilon(epsilon)
	svm.parallel.set_num_threads(num_threads)
	svm.set_bias_enabled(False)
	svm.set_max_train_time(max_train_time)
	svm.train()

	svm.set_features(feats_test)
	svm.classify().get_labels()

def svmocas ():
	print 'SVMOcas'

	num_feats=11
	data=get_clouds(2, num_feats, 12)
	realfeat=RealFeatures(data)
	feats_test=SparseRealFeatures()
	feats_test.obtain_from_simple(realfeat)
	data=get_clouds(2, num_feats, 21)
	realfeat=RealFeatures(data)
	feats_test=SparseRealFeatures()
	feats_test.obtain_from_simple(realfeat)

	C=0.42
	epsilon=1e-5
	num_threads=1
	lab=rand(feats_test.get_num_vectors()).round()*2-1
	labels=Labels(array(lab))

	svm=SVMOcas(C, feats_test, labels)
	svm.set_epsilon(epsilon)
	svm.parallel.set_num_threads(num_threads)
	svm.set_bias_enabled(False)
	svm.train()

	svm.set_features(feats_test)
	svm.classify().get_labels()

def liblinear ():
	print 'LibLinear'

	num_feats=11
	data=get_clouds(2, num_feats, 10)
	realfeat=RealFeatures(data)
	feats_train=SparseRealFeatures()
	feats_train.obtain_from_simple(realfeat)
	data=get_clouds(2, num_feats, 25)
	realfeat=RealFeatures(data)
	feats_test=SparseRealFeatures()
	feats_test.obtain_from_simple(realfeat)

	C=0.42
	epsilon=1e-5
	num_threads=1
	lab=rand(feats_train.get_num_vectors()).round()*2-1
	labels=Labels(array(lab))

	svm=LibLinear(C, feats_train, labels)
	svm.set_epsilon(epsilon)
	svm.parallel.set_num_threads(num_threads)
	svm.set_bias_enabled(True)
	svm.train()

	svm.set_features(feats_test)
	svm.classify().get_labels()

def svmlin ():
	print 'SVMLin'

	num_feats=11
	data=get_clouds(2, num_feats, 11)
	realfeat=RealFeatures(data)
	feats_train=SparseRealFeatures()
	feats_train.obtain_from_simple(realfeat)
	data=get_clouds(2, num_feats, 25)
	realfeat=RealFeatures(data)
	feats_test=SparseRealFeatures()
	feats_test.obtain_from_simple(realfeat)

	C=0.42
	epsilon=1e-5
	num_threads=1
	lab=rand(feats_train.get_num_vectors()).round()*2-1
	labels=Labels(array(lab))

	svm=SVMLin(C, feats_train, labels)
	svm.set_epsilon(epsilon)
	svm.parallel.set_num_threads(num_threads)
	svm.set_bias_enabled(True)
	svm.train()

	svm.set_features(feats_test)
	svm.get_bias()
	svm.get_w()
	svm.classify().get_labels()

###########################################################################
# misc classifiers
###########################################################################

def perceptron ():
	print 'Perceptron'

	num_feats=9
	data=get_clouds(2, num_feats, 11)
	feats_train=RealFeatures(data)
	data=get_clouds(2, num_feats, 17)
	feats_test=RealFeatures(data)

	learn_rate=1.
	max_iter=1000
	num_threads=1
	lab=rand(feats_train.get_num_vectors()).round()*2-1
	labels=Labels(array(lab))

	perceptron=Perceptron(feats_train, labels)
	perceptron.set_learn_rate(learn_rate)
	perceptron.set_max_iter(max_iter)
	perceptron.parallel.set_num_threads(num_threads)
	perceptron.train()

	perceptron.set_features(feats_test)
	perceptron.classify().get_labels()

def knn ():
	print 'KNN'

	num_feats=9
	data=get_clouds(2, num_feats, 10)
	feats_train=RealFeatures(data)
	data=get_clouds(2, num_feats, 18)
	feats_test=RealFeatures(data)
	distance=EuclidianDistance()

	k=3
	num_threads=1
	lab=rand(feats_train.get_num_vectors()).round()*2-1
	labels=Labels(array(lab))

	knn=KNN(k, distance, labels)
	knn.parallel.set_num_threads(num_threads)
	knn.train()

	distance.init(feats_train, feats_test)
	knn.classify().get_labels()

def lda ():
	print 'LDA'

	num_feats=9
	data=get_clouds(2, num_feats, 14)
	feats_train=RealFeatures(data)
	data=get_clouds(2, num_feats, 27)
	feats_test=RealFeatures(data)

	gamma=3
	num_threads=1
	lab=rand(feats_train.get_num_vectors()).round()*2-1
	labels=Labels(array(lab))

	lda=LDA(gamma, feats_train, labels)
	lda.parallel.set_num_threads(num_threads)
	lda.train()

	lda.get_bias()
	lda.get_w()
	lda.set_features(feats_test)
	lda.classify().get_labels()

###########################################################################
# call functions
###########################################################################

if __name__=='__main__':
	seed(42)

	svm_light()
	libsvm()
	gpbtsvm()
	mpdsvm()
	libsvm_multiclass()
	libsvm_oneclass()
	gmnpsvm()

	do_batch_linadd()

	subgradient_svm()
	svmocas()
	liblinear()
	svmlin()

	perceptron()
	knn()
	lda()
