#!/usr/bin/env python
import numpy
from modshogun import RealFeatures, MSG_DEBUG

numpy.random.seed(17)
traindat = numpy.random.random_sample((10,10))
testdat = numpy.random.random_sample((10,10))
parameter_list=[[traindat,testdat,1.2],[traindat,testdat,1.4]]

def distance_director_euclidean_modular (fm_train_real=traindat,fm_test_real=testdat,scale=1.2):
	try:
		from modshogun import DirectorDistance
	except ImportError:
		print("recompile shogun with --enable-swig-directors")
		return

	class DirectorEuclideanDistance(DirectorDistance):
		def __init__(self):
			DirectorDistance.__init__(self, True)
		def distance_function(self, idx_a, idx_b):
			seq1 = self.get_lhs().get_feature_vector(idx_a)
			seq2 = self.get_rhs().get_feature_vector(idx_b)
			return numpy.linalg.norm(seq1-seq2)

	from modshogun import EuclideanDistance
	from modshogun import Time

	feats_train=RealFeatures(fm_train_real)
	#feats_train.io.set_loglevel(MSG_DEBUG)
	feats_train.parallel.set_num_threads(1)
	feats_test=RealFeatures(fm_test_real)

	distance=EuclideanDistance()
	distance.init(feats_train, feats_test)

	ddistance=DirectorEuclideanDistance()
	ddistance.init(feats_train, feats_test)

	#print  "dm_train"
	t=Time()
	dm_train=distance.get_distance_matrix()
	#t1=t.cur_time_diff(True)

	#print  "ddm_train"
	t=Time()
	ddm_train=ddistance.get_distance_matrix()
	#t2=t.cur_time_diff(True)	

	#print "dm_train", dm_train
	#print "ddm_train", ddm_train

	return dm_train, ddm_train

if __name__=='__main__':
	print('DirectorEuclideanDistance')
	distance_director_euclidean_modular(*parameter_list[0])
