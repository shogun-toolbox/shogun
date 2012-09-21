#!/usr/bin/env python
#!/usr/bin/env perl
#
# This program is free software you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation either version 3 of the License, or
# (at your option) any later version.
#
# Written (C) 2012 Heiko Strathmann
#
from numpy import *
#from matplotlib import pyplot

# performs learning of optimal non-negative kernel weights for a linear time
# two sample test using the linear time Maximum Mean Discrepancy
def statistics_linear_time_mmd_kernel_choice ():
	from shogun.Features import RealFeatures, CombinedFeatures
	from shogun.Features import DataGenerator
	from shogun.Kernel import GaussianKernel, CombinedKernel
	from shogun.Statistics import LinearTimeMMD
	from shogun.Statistics import BOOTSTRAP, MMD1_GAUSSIAN

	# note that the linear time statistic is designed for much larger datasets
	n=50000
	dim=5
	difference=2

	# use data generator class to produce example data
	# in pratice, this generate data function could be replaced by a method
	# that obtains data from a stream
	data=DataGenerator.generate_mean_data(n,dim,difference)
	
	print "dimension means of X", mean(data.T[0:n].T)
	print "dimension means of Y", mean(data.T[n:2*n+1].T)

	# create kernels/features to choose from
	# here: just a bunch of Gaussian Kernels with different widths
	# real sigmas are 2^-5, ..., 2^10
	sigmas=array([pow(2,x) for x in range(-5,10)])
	
	# shogun has a different parametrization of the Gaussian kernel
	shogun_sigmas=array([x*x*2 for x in sigmas])
	
	# We will use multiple kernels
	kernel=CombinedKernel()
	
	# two separate feature objects here, could also be one with appended data
	features=CombinedFeatures()
	
	# all kernels work on same features
	for i in range(len(sigmas)):
		kernel.append_kernel(GaussianKernel(10, shogun_sigmas[i]))
		features.append_feature_obj(RealFeatures(data))
	
	mmd=LinearTimeMMD(kernel,features, n)
	
	print "start learning kernel weights"
	mmd.set_opt_regularization_eps(10E-5)
	mmd.set_opt_low_cut(10E-5)
	mmd.set_opt_max_iterations(1000)
	mmd.set_opt_epsilon(10E-7)
	mmd.optimize_kernel_weights()
	weights=kernel.get_subkernel_weights()
	print "learned weights:", weights
	#pyplot.plot(array(range(len(sigmas))), weights)
	#pyplot.show()
	print "index of max weight", weights.argmax()

if __name__=='__main__':
	print('LinearTimeMMDKernelChoice')
	statistics_linear_time_mmd_kernel_choice()
