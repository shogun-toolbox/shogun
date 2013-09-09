#!/usr/bin/env python

from numpy import random
from modshogun import LibSVM
from modshogun import RealFeatures, Labels
from modshogun import LinearKernel


num_feats=23
num_vec=42

scale=2.1
size_cache=10

C=0.017
epsilon=1e-5
tube_epsilon=1e-2
svm=LibSVM()
svm.set_C(C, C)
svm.set_epsilon(epsilon)
svm.set_tube_epsilon(tube_epsilon)

for i in range(3):
	data_train=random.rand(num_feats, num_vec)
	data_test=random.rand(num_feats, num_vec)
	feats_train=RealFeatures(data_train)
	feats_test=RealFeatures(data_test)
	labels=Labels(random.rand(num_vec).round()*2-1)

	svm.set_kernel(LinearKernel(size_cache, scale))
	svm.set_labels(labels)

	kernel=svm.get_kernel()
	print("kernel cache size: %s" % (kernel.get_cache_size()))

	kernel.init(feats_test, feats_test)
	svm.train()

	kernel.init(feats_train, feats_test)
	print(svm.apply().get_labels())

	#kernel.remove_lhs_and_rhs()

	#import pdb
	#pdb.set_trace()

