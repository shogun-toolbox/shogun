import numpy
from shogun.Features import RealFeatures
try:
	from shogun.Distance import DirectorDistance
except ImportError:
	print "recompile shogun with --enable-swig-directors"
	import sys
	sys.exit(0)


class DirectorEuclidianDistance(DirectorDistance):
	def __init__(self):
		DirectorDistance.__init__(self, True)
	def distance_function(self, idx_a, idx_b):
		seq1 = self.get_lhs().get_feature_vector(idx_a)
               	seq2 = self.get_rhs().get_feature_vector(idx_b)
		return numpy.linalg.norm(seq1-seq2)

traindat = numpy.random.random_sample((10,10))
testdat = numpy.random.random_sample((10,10))
parameter_list=[[traindat,testdat,1.2],[traindat,testdat,1.4]]

def distance_director_euclidian_modular (fm_train_real=traindat,fm_test_real=testdat,scale=1.2):

	from shogun.Distance import EuclidianDistance
	from modshogun import Time

	feats_train=RealFeatures(fm_train_real)
	feats_train.io.set_loglevel(0)
	feats_train.parallel.set_num_threads(1)
	feats_test=RealFeatures(fm_test_real)

	distance=EuclidianDistance()
	distance.init(feats_train, feats_test)

	ddistance=DirectorEuclidianDistance()
	ddistance.init(feats_train, feats_test)

	print  "dm_train"
	t=Time()
	dm_train=distance.get_distance_matrix()
	t1=t.cur_time_diff(True)

	print  "ddm_train"
	t=Time()
	ddm_train=ddistance.get_distance_matrix()
	t2=t.cur_time_diff(True)	

	print "dm_train", dm_train
	print "ddm_train", ddm_train

	return dm_train, ddm_train

if __name__=='__main__':
	print('DirectorEuclidianDistance')
	distance_director_euclidian_modular(*parameter_list[0])
