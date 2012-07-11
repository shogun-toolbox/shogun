#
# This program is free software you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation either version 3 of the License, or
# (at your option) any later version.
#
# Written (C) 2012 Heiko Strathmann
#
from numpy import *
from tools.two_distributions_data import TwoDistributionsData

gen_data=TwoDistributionsData()

# performs learning of optimal non-negative kernel weights for a linear time
# two sample test using the linear time Maximum Mean Discrepancy
def statistics_linear_time_mmd_kernel_choice():
	from shogun.Features import RealFeatures, CombinedFeatures
	from shogun.Kernel import GaussianKernel, CombinedKernel
	from shogun.Statistics import LinearTimeMMD
	from shogun.Statistics import BOOTSTRAP, MMD1_GAUSSIAN

	# note that the linear time statistic is designed for much larger datasets
	n=50000
	dim=5
	difference=2

	# data is standard normal distributed. only one dimension of Y has a mean
	# shift of difference
	# in pratice, this generate data function could be replaced by a method
	# that obtains data from a stream
	(X,Y)=gen_data.create_mean_data(n,dim,difference)
	
	# concatenate since MMD class takes data as one feature object
	# (it is possible to give two, but then data is copied)
	Z=concatenate((X,Y), axis=1)
	print "dimension means of X", [mean(x) for x in X]
	print "dimension means of Y", [mean(x) for x in Y]

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
	
	for i in range(len(sigmas)):
		kernel.append_kernel(GaussianKernel(10, shogun_sigmas[i]))
		features.append_feature_obj(RealFeatures(Z))
	
	mmd=LinearTimeMMD(kernel,features, n)
	
	print "start learning kernel weights"
	#mmd.set_log_level(MSG_DEBUG)
	mmd.optimize_kernel_weights()
	print "learned weights:", kernel.get_subkernel_weights()

if __name__=='__main__':
	print('LinearTimeMMDKernelChoice')
	statistics_linear_time_mmd_kernel_choice()
