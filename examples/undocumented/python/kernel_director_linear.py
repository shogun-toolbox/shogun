#!/usr/bin/env python
import numpy as np

traindat = np.random.random_sample((10,10))
testdat = np.random.random_sample((10,10))
parameter_list=[[traindat,testdat,1.2],[traindat,testdat,1.4]]

def kernel_director_linear (fm_train_real=traindat,fm_test_real=testdat,scale=1.2):
	try:
		from shogun import DirectorKernel
	except ImportError:
		print("recompile shogun with --enable-swig-directors")
		return
	import shogun as sg

	class DirectorLinearKernel(DirectorKernel):
		def __init__(self):
			DirectorKernel.__init__(self, True)
		def kernel_function(self, idx_a, idx_b):
			seq1 = self.get_lhs().get("feature_matrix")[idx_a]
			seq2 = self.get_rhs().get("feature_matrix")[idx_b]
			return np.dot(seq1, seq2)


	from shogun import Time

	feats_train=sg.create_features(fm_train_real)
	#feats_train.io.set_loglevel(0)
	feats_train.get_global_parallel().set_num_threads(1)
	feats_test=sg.create_features(fm_test_real)

	kernel=sg.create_kernel("LinearKernel")
	kernel.set_normalizer(sg.create_kernel_normalizer("AvgDiagKernelNormalizer", scale=scale))
	kernel.init(feats_train, feats_train)

	dkernel=DirectorLinearKernel()
	dkernel.set_normalizer(sg.create_kernel_normalizer("AvgDiagKernelNormalizer", scale=scale))
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
	kernel_director_linear(*parameter_list[0])
