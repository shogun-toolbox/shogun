#!/usr/bin/env python
"""
Explicit examples on how to use the different classifiers
"""

from numpy import double, array, char, floor, concatenate
from numpy.random import rand, seed, permutation
from shogun.Kernel import GaussianKernel, WeightedDegreeStringKernel
from shogun.Distance import EuclidianDistance
from shogun.Features import *
from shogun.Classifier import *

from tools.load import LoadMatrix
lm=LoadMatrix()
fm_train_real=lm.load_numbers('../data/fm_train_real.dat')
fm_test_real=lm.load_numbers('../data/fm_test_real.dat')
fm_train_dna=lm.load_dna('../data/fm_train_dna.dat')
fm_test_dna=lm.load_dna('../data/fm_test_dna.dat')
label_train_dna=lm.load_labels('../data/label_train_dna.dat')
label_train_twoclass=lm.load_labels('../data/label_train_twoclass.dat')
label_train_multiclass=lm.load_labels('../data/label_train_multiclass.dat')

###########################################################################
# kernel-based SVMs
###########################################################################

def svm_light ():
	print 'SVMLight'

	feats_train=StringCharFeatures(DNA)
	feats_train.set_string_features(fm_train_dna)
	feats_test=StringCharFeatures(DNA)
	feats_test.set_string_features(fm_test_dna)
	degree=20

	kernel=WeightedDegreeStringKernel(feats_train, feats_train, degree)

	C=0.017
	epsilon=1e-5
	tube_epsilon=1e-2
	num_threads=1
	labels=Labels(label_train_dna)

	try:
		svm=SVMLight(C, kernel, labels)
	except NameError:
		print 'No support for SVMLight available.'
		return

	svm.set_epsilon(epsilon)
	svm.set_tube_epsilon(tube_epsilon)
	svm.parallel.set_num_threads(num_threads)
	svm.train()

	kernel.init(feats_train, feats_test)
	svm.classify().get_labels()

def libsvm ():
	print 'LibSVM'

	feats_train=RealFeatures(fm_train_real)
	feats_test=RealFeatures(fm_test_real)
	width=2.1
	kernel=GaussianKernel(feats_train, feats_train, width)

	C=0.017
	epsilon=1e-5
	tube_epsilon=1e-2
	num_threads=2
	labels=Labels(label_train_twoclass)

	svm=LibSVM(C, kernel, labels)
	svm.set_epsilon(epsilon)
	svm.set_tube_epsilon(tube_epsilon)
	svm.parallel.set_num_threads(num_threads)
	svm.train()

	kernel.init(feats_train, feats_test)
	svm.classify().get_labels()

def gpbtsvm ():
	print 'GPBTSVM'

	feats_train=RealFeatures(fm_train_real)
	feats_test=RealFeatures(fm_test_real)
	width=2.1
	kernel=GaussianKernel(feats_train, feats_train, width)

	C=0.017
	epsilon=1e-5
	tube_epsilon=1e-2
	num_threads=8
	labels=Labels(label_train_twoclass)

	svm=GPBTSVM(C, kernel, labels)
	svm.set_epsilon(epsilon)
	svm.set_tube_epsilon(tube_epsilon)
	svm.parallel.set_num_threads(num_threads)
	svm.train()

	kernel.init(feats_train, feats_test)
	svm.classify().get_labels()

def mpdsvm ():
	print 'MPDSVM'

	feats_train=RealFeatures(fm_train_real)
	feats_test=RealFeatures(fm_test_real)
	width=2.1
	kernel=GaussianKernel(feats_train, feats_train, width)

	C=0.017
	epsilon=1e-5
	tube_epsilon=1e-2
	num_threads=1
	labels=Labels(label_train_twoclass)

	svm=MPDSVM(C, kernel, labels)
	svm.set_epsilon(epsilon)
	svm.set_tube_epsilon(tube_epsilon)
	svm.parallel.set_num_threads(num_threads)
	svm.train()

	kernel.init(feats_train, feats_test)
	svm.classify().get_labels()

def libsvm_multiclass ():
	print 'LibSVMMultiClass'

	feats_train=RealFeatures(fm_train_real)
	feats_test=RealFeatures(fm_test_real)
	width=2.1
	kernel=GaussianKernel(feats_train, feats_train, width)

	C=0.017
	epsilon=1e-5
	tube_epsilon=1e-2
	num_threads=8
	labels=Labels(label_train_multiclass)

	svm=LibSVMMultiClass(C, kernel, labels)
	svm.set_epsilon(epsilon)
	svm.set_tube_epsilon(tube_epsilon)
	svm.parallel.set_num_threads(num_threads)
	svm.train()

	kernel.init(feats_train, feats_test)
	svm.classify().get_labels()

def libsvm_oneclass ():
	print 'LibSVMOneClass'

	feats_train=RealFeatures(fm_train_real)
	feats_test=RealFeatures(fm_test_real)
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

	feats_train=RealFeatures(fm_train_real)
	feats_test=RealFeatures(fm_test_real)
	width=2.1
	kernel=GaussianKernel(feats_train, feats_train, width)

	C=0.017
	epsilon=1e-5
	tube_epsilon=1e-2
	num_threads=1
	labels=Labels(label_train_multiclass)

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

	feats_train=StringCharFeatures(DNA)
	feats_train.set_string_features(fm_train_dna)
	feats_test=StringCharFeatures(DNA)
	feats_test.set_string_features(fm_test_dna)
	degree=20

	kernel=WeightedDegreeStringKernel(feats_train, feats_train, degree)

	C=0.017
	epsilon=1e-5
	tube_epsilon=1e-2
	num_threads=2
	labels=Labels(label_train_dna)

	svm=LibSVM(C, kernel, labels)
	svm.set_epsilon(epsilon)
	svm.set_tube_epsilon(tube_epsilon)
	svm.parallel.set_num_threads(num_threads)
	svm.train()

	kernel.init(feats_train, feats_test)

	#print 'LibSVM Objective: %f num_sv: %d' % \
	#	(svm.get_objective(), svm.get_num_support_vectors())
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

	realfeat=RealFeatures(fm_train_real)
	feats_train=SparseRealFeatures()
	feats_train.obtain_from_simple(realfeat)
	realfeat=RealFeatures(fm_test_real)
	feats_test=SparseRealFeatures()
	feats_test.obtain_from_simple(realfeat)

	C=0.42
	epsilon=1e-3
	num_threads=1
	max_train_time=1.
	labels=Labels(label_train_twoclass)

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

	realfeat=RealFeatures(fm_train_real)
	feats_train=SparseRealFeatures()
	feats_train.obtain_from_simple(realfeat)
	realfeat=RealFeatures(fm_test_real)
	feats_test=SparseRealFeatures()
	feats_test.obtain_from_simple(realfeat)

	C=0.42
	epsilon=1e-5
	num_threads=1
	labels=Labels(label_train_twoclass)

	svm=SVMOcas(C, feats_train, labels)
	svm.set_epsilon(epsilon)
	svm.parallel.set_num_threads(num_threads)
	svm.set_bias_enabled(False)
	svm.train()

	svm.set_features(feats_test)
	svm.classify().get_labels()

def svmsgd ():
	print 'SVMSGD'

	realfeat=RealFeatures(fm_train_real)
	feats_train=SparseRealFeatures()
	feats_train.obtain_from_simple(realfeat)
	realfeat=RealFeatures(fm_test_real)
	feats_test=SparseRealFeatures()
	feats_test.obtain_from_simple(realfeat)

	C=0.42
	epsilon=1e-5
	num_threads=1
	labels=Labels(label_train_twoclass)

	svm=SVMSGD(C, feats_train, labels)
	#svm.io.set_loglevel(0)
	svm.train()

	svm.set_features(feats_test)
	svm.classify().get_labels()

def liblinear ():
	print 'LibLinear'

	realfeat=RealFeatures(fm_train_real)
	feats_train=SparseRealFeatures()
	feats_train.obtain_from_simple(realfeat)
	realfeat=RealFeatures(fm_test_real)
	feats_test=SparseRealFeatures()
	feats_test.obtain_from_simple(realfeat)

	C=0.42
	epsilon=1e-5
	num_threads=1
	labels=Labels(label_train_twoclass)

	svm=LibLinear(C, feats_train, labels)
	svm.set_epsilon(epsilon)
	svm.parallel.set_num_threads(num_threads)
	svm.set_bias_enabled(True)
	svm.train()

	svm.set_features(feats_test)
	svm.classify().get_labels()

def svmlin ():
	print 'SVMLin'

	realfeat=RealFeatures(fm_train_real)
	feats_train=SparseRealFeatures()
	feats_train.obtain_from_simple(realfeat)
	realfeat=RealFeatures(fm_test_real)
	feats_test=SparseRealFeatures()
	feats_test.obtain_from_simple(realfeat)

	C=0.42
	epsilon=1e-5
	num_threads=1
	labels=Labels(label_train_twoclass)

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

	feats_train=RealFeatures(fm_train_real)
	feats_test=RealFeatures(fm_test_real)

	learn_rate=1.
	max_iter=1000
	num_threads=1
	labels=Labels(label_train_twoclass)

	perceptron=Perceptron(feats_train, labels)
	perceptron.set_learn_rate(learn_rate)
	perceptron.set_max_iter(max_iter)
	perceptron.parallel.set_num_threads(num_threads)
	# often does not converge
	#perceptron.train()

	#perceptron.set_features(feats_test)
	#perceptron.classify().get_labels()

def knn ():
	print 'KNN'

	feats_train=RealFeatures(fm_train_real)
	feats_test=RealFeatures(fm_test_real)
	distance=EuclidianDistance()

	k=3
	num_threads=1
	labels=Labels(label_train_twoclass)

	knn=KNN(k, distance, labels)
	knn.parallel.set_num_threads(num_threads)
	knn.train()

	distance.init(feats_train, feats_test)
	knn.classify().get_labels()

def lda ():
	print 'LDA'

	feats_train=RealFeatures(fm_train_real)
	feats_test=RealFeatures(fm_test_real)

	gamma=3
	num_threads=1
	labels=Labels(label_train_twoclass)

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

	svmsgd()
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
