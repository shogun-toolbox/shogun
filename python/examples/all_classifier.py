#!/usr/bin/env python
"""
Explicit examples on how to use the different classifiers
"""

from numpy import double, array, floor, concatenate, sign, ones, zeros
from numpy.random import rand, seed, permutation
from sg import sg

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
	size_cache=10
	degree=20
	C=0.017
	epsilon=1e-5
	use_bias=False
	num_feats=14
	num_trainvec=11
	labels=sign(rand(1, num_trainvec)-0.5)[0]

	sg('set_features', 'TRAIN', data['train'], 'DNA')
	sg('set_kernel', 'WEIGHTEDDEGREE', 'CHAR', size_cache, degree)
	sg('init_kernel', 'TRAIN')

	sg('set_labels', 'TRAIN', labels)
	sg('new_svm', 'LIGHT')
	sg('svm_epsilon', epsilon)
	sg('c', C)
	sg('svm_use_bias', use_bias)
	sg('svm_train')

	sg('set_features', 'TEST', data['test'], 'DNA')
	sg('init_kernel', 'TEST')
	result=sg('svm_classify')

def libsvm ():
	print 'LibSVM'

	size_cache=10
	width=2.1
	C=0.017
	epsilon=1e-5
	use_bias=False
	num_feats=11
	num_trainvec=12

	trainlab=sign(rand(1, num_trainvec*2)-0.5)[0]
	traindata=get_clouds(2, num_feats, num_trainvec)
	testdata=get_clouds(2, num_feats, 17)

	sg('set_features', 'TRAIN', traindata)
	sg('set_kernel', 'GAUSSIAN', 'REAL', size_cache, width)
	sg('init_kernel', 'TRAIN')

	sg('set_labels', 'TRAIN', trainlab)
	sg('new_svm', 'LIBSVM')
	sg('svm_epsilon', epsilon)
	sg('c', C)
	sg('svm_use_bias', use_bias)
	sg('svm_train')

	sg('set_features', 'TEST', testdata)
	sg('init_kernel', 'TEST')
	result=sg('svm_classify')

def gpbtsvm ():
	print 'GPBTSVM'

	size_cache=10
	width=2.1
	C=0.017
	epsilon=1e-5
	use_bias=False
	num_feats=9
	num_trainvec=11

	trainlab=sign(rand(1, num_trainvec*2)-0.5)[0]
	traindata=get_clouds(2, num_feats, num_trainvec)
	testdata=get_clouds(2, num_feats, 16)

	sg('set_features', 'TRAIN', traindata)
	sg('set_kernel', 'GAUSSIAN', 'REAL', size_cache, width)
	sg('init_kernel', 'TRAIN')

	sg('set_labels', 'TRAIN', trainlab)
	sg('new_svm', 'GPBTSVM')
	sg('svm_epsilon', epsilon)
	sg('c', C)
	sg('svm_use_bias', use_bias)
	sg('svm_train')

	sg('set_features', 'TEST', testdata)
	sg('init_kernel', 'TEST')
	result=sg('svm_classify')

def mpdsvm ():
	print 'MPDSVM'

	size_cache=10
	width=2.1
	C=0.017
	epsilon=1e-5
	use_bias=False
	num_feats=13
	num_trainvec=11

	trainlab=sign(rand(1, num_trainvec*2)-0.5)[0]
	traindata=get_clouds(2, num_feats, num_trainvec)
	testdata=get_clouds(2, num_feats, 18)

	sg('set_features', 'TRAIN', traindata)
	sg('set_kernel', 'GAUSSIAN', 'REAL', size_cache, width)
	sg('init_kernel', 'TRAIN')

	sg('set_labels', 'TRAIN', trainlab)
	sg('new_svm', 'MPDSVM')
	sg('svm_epsilon', epsilon)
	sg('c', C)
	sg('svm_use_bias', use_bias)
	sg('svm_train')

	sg('set_features', 'TEST', testdata)
	sg('init_kernel', 'TEST')
	result=sg('svm_classify')

def libsvm_multiclass ():
	print 'LibSVMMultiClass'

	size_cache=10
	width=2.1
	C=10.
	epsilon=1e-5
	use_bias=False
	num_feats=9
	num_trainvec=11

	trainlab=array([double(x) for x in xrange(num_trainvec*2)])
	traindata=get_clouds(2, num_feats, num_trainvec)
	testdata=get_clouds(2, num_feats, 23)

	sg('set_features', 'TRAIN', traindata)
	sg('set_kernel', 'GAUSSIAN', 'REAL', size_cache, width)
	sg('init_kernel', 'TRAIN')

	sg('set_labels', 'TRAIN', trainlab)
	sg('new_svm', 'LIBSVM_MULTICLASS')
	sg('svm_epsilon', epsilon)
	sg('c', C)
	sg('svm_use_bias', use_bias)
	sg('svm_train')

	sg('set_features', 'TEST', testdata)
	sg('init_kernel', 'TEST')
	result=sg('svm_classify')

def libsvm_oneclass ():
	print 'LibSVMOneClass'

	size_cache=10
	width=2.1
	C=10.
	epsilon=1e-5
	use_bias=False
	num_feats=9
	num_trainvec=11

	traindata=get_clouds(2, num_feats, num_trainvec)
	testdata=get_clouds(2, num_feats, 23)

	sg('set_features', 'TRAIN', traindata)
	sg('set_kernel', 'GAUSSIAN', 'REAL', size_cache, width)
	sg('init_kernel', 'TRAIN')

	sg('new_svm', 'LIBSVM_ONECLASS')
	sg('svm_epsilon', epsilon)
	sg('c', C)
	sg('svm_use_bias', use_bias)
	sg('svm_train')

	sg('set_features', 'TEST', testdata)
	sg('init_kernel', 'TEST')
	result=sg('svm_classify')

def gmnpsvm ():
	print 'GMNPSVM'

	size_cache=10
	width=2.1
	C=0.017
	epsilon=1e-5
	use_bias=False
	num_feats=32
	num_trainvec=12

	trainlab=sign(rand(1, num_trainvec*2)-0.5)[0]
	traindata=get_clouds(2, num_feats, num_trainvec)
	testdata=get_clouds(2, num_feats, 18)

	sg('set_features', 'TRAIN', traindata)
	sg('set_kernel', 'GAUSSIAN', 'REAL', size_cache, width)
	sg('init_kernel', 'TRAIN')

	sg('set_labels', 'TRAIN', trainlab)
	sg('new_svm', 'GMNPSVM')
	sg('svm_epsilon', epsilon)
	sg('c', C)
	sg('svm_use_bias', use_bias)
	sg('svm_train')

	sg('set_features', 'TEST', testdata)
	sg('init_kernel', 'TEST')
	result=sg('svm_classify')

###########################################################################
# run with batch or linadd on LibSVM
###########################################################################

def do_batch_linadd ():
	print 'LibSVM batch'

	size_cache=10
	width=2.1
	C=0.017
	epsilon=1e-5
	use_bias=False
	num_feats=11
	num_trainvec=12

	trainlab=sign(rand(1, num_trainvec*2)-0.5)[0]
	traindata=get_clouds(2, num_feats, num_trainvec)
	testdata=get_clouds(2, num_feats, 17)

	sg('set_features', 'TRAIN', traindata)
	sg('set_kernel', 'GAUSSIAN', 'REAL', size_cache, width)
	sg('init_kernel', 'TRAIN')

	sg('set_labels', 'TRAIN', trainlab)
	sg('new_svm', 'LIBSVM')
	sg('svm_epsilon', epsilon)
	sg('c', C)
	sg('svm_use_bias', use_bias)
	sg('svm_train')

	sg('set_features', 'TEST', testdata)
	sg('init_kernel', 'TEST')

	objective=sg('get_svm_objective')
	sg('use_batch_computation', True)
	sg('use_linadd', True)
	result=sg('svm_classify')

###########################################################################
# misc classifiers
###########################################################################

def perceptron ():
	print 'Perceptron'

	num_feats=14
	num_trainvec=10

	trainlab=sign(rand(1, num_trainvec*2)-0.5)[0]
	traindata=get_clouds(2, num_feats, num_trainvec)
	testdata=get_clouds(2, num_feats, 42)

	sg('set_features', 'TRAIN', traindata)
	sg('set_labels', 'TRAIN', trainlab)
	sg('new_classifier', 'PERCEPTRON')
	sg('train_classifier')

	sg('set_features', 'TEST', testdata)
	result=sg('classify')

def knn ():
	print 'KNN'

	num_feats=14
	num_trainvec=10
	k=3

	trainlab=sign(rand(1, num_trainvec*2)-0.5)[0]
	traindata=get_clouds(2, num_feats, num_trainvec)
	testdata=get_clouds(2, num_feats, 42)

	sg('set_features', 'TRAIN', traindata)
	sg('set_labels', 'TRAIN', trainlab)
	sg('set_distance', 'EUCLIDIAN', 'REAL')
	sg('init_distance', 'TRAIN')
	sg('new_classifier', 'KNN')
	sg('train_classifier', k)

	sg('set_features', 'TEST', testdata)
	sg('init_distance', 'TEST')
	result=sg('classify')

def lda ():
	print 'LDA'

	num_feats=14
	num_trainvec=10

	trainlab=sign(rand(1, num_trainvec*2)-0.5)[0]
	traindata=get_clouds(2, num_feats, num_trainvec)
	testdata=get_clouds(2, num_feats, 42)

	sg('set_features', 'TRAIN', traindata)
	sg('set_labels', 'TRAIN', trainlab)
	sg('new_classifier', 'LDA')
	sg('train_classifier')

	sg('set_features', 'TEST', testdata)
	result=sg('classify')

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

	perceptron()
	knn()
	lda()
