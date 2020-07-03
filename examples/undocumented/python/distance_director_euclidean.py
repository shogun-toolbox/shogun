#!/usr/bin/env python
import numpy
import time
from shogun import MSG_DEBUG

numpy.random.seed(17)
traindat = numpy.random.random_sample((10,10))
testdat = numpy.random.random_sample((10,10))
parameter_list=[[traindat,testdat,1.2],[traindat,testdat,1.4]]

def distance_director_euclidean (fm_train_real=traindat,fm_test_real=testdat,scale=1.2):
	try:
		from shogun import DirectorDistance
	except ImportError:
		print("recompile shogun with --enable-swig-directors")
		return
	import shogun as sg

	class DirectorEuclideanDistance(DirectorDistance):
		def __init__(self):
			DirectorDistance.__init__(self, True)
		def distance_function(self, idx_a, idx_b):
			seq1 = self.get_lhs().get_feature_vector(idx_a)
			seq2 = self.get_rhs().get_feature_vector(idx_b)
			return numpy.linalg.norm(seq1-seq2)

	feats_train=sg.create_features(fm_train_real)
	#feats_train.io.set_loglevel(MSG_DEBUG)
	feats_train.get_global_parallel().set_num_threads(1)
	feats_test=sg.create_features(fm_test_real)

	distance=sg.create_distance("EuclideanDistance")
	distance.init(feats_train, feats_test)

	ddistance=DirectorEuclideanDistance()
	ddistance.init(feats_train, feats_test)

	#print  "dm_train"
	t=time.perf_counter()
	dm_train=distance.get_distance_matrix()
	t1=time.perf_counter() - t

	#print  "ddm_train"
	t=time.perf_counter()
	ddm_train=ddistance.get_distance_matrix()
	t2=time.perf_counter() - t

	#print "dm_train", dm_train
	#print "ddm_train", ddm_train

	return dm_train, ddm_train

if __name__=='__main__':
	print('DirectorEuclideanDistance')
	distance_director_euclidean(*parameter_list[0])
