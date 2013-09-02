#!/usr/bin/env python
import numpy
from modshogun import RealFeatures, MSG_DEBUG
traindat = numpy.random.random_sample((10,10))
testdat = numpy.random.random_sample((10,10))
parameter_list=[[traindat,testdat,1.2],[traindat,testdat,1.4]]

def kernel_director_linear_modular (fm_train_real=traindat,fm_test_real=testdat,scale=1.2):
	try:
		from modshogun import DirectorKernel
	except ImportError:
		print("recompile shogun with --enable-swig-directors")
		return

	class DirectorLinearKernel(DirectorKernel):
		def __init__(self):
			DirectorKernel.__init__(self, True)
		def kernel_function(self, idx_a, idx_b):
			seq1 = self.get_lhs().get_feature_vector(idx_a)
			seq2 = self.get_rhs().get_feature_vector(idx_b)
			return numpy.dot(seq1, seq2)


	from modshogun import LinearKernel, AvgDiagKernelNormalizer
	from modshogun import Time

	feats_train=RealFeatures(fm_train_real)
	#feats_train.io.set_loglevel(MSG_DEBUG)
	feats_train.parallel.set_num_threads(1)
	feats_test=RealFeatures(fm_test_real)
	 
	kernel=LinearKernel()
	kernel.set_normalizer(AvgDiagKernelNormalizer(scale))
	kernel.init(feats_train, feats_train)

	dkernel=DirectorLinearKernel()
	dkernel.set_normalizer(AvgDiagKernelNormalizer(scale))
	dkernel.init(feats_train, feats_train)

	#print  "km_train"
	t=Time()
	km_train=kernel.get_kernel_matrix()
	#t1=t.cur_time_diff(True)

	#print  "dkm_train"
	t=Time()
	dkm_train=dkernel.get_kernel_matrix()
	#t2=t.cur_time_diff(True)

	#print "km_train", km_train
	#print "dkm_train", dkm_train

	return km_train, dkm_train

if __name__=='__main__':
	print('DirectorLinear')
	kernel_director_linear_modular(*parameter_list[0])
