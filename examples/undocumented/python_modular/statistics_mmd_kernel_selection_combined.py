#!/usr/bin/env python
#
# This program is free software you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation either version 3 of the License, or
# (at your option) any later version.
#
# Written (C) 2012-2013 Heiko Strathmann
#
from numpy import *
#from pylab import *

parameter_list = [[1000,10,5,3,pi/4, "opt"], [1000,10,5,3,pi/4, "l2"]]


def statistics_mmd_kernel_selection_combined(m,distance,stretch,num_blobs,angle,selection_method):
	from modshogun import RealFeatures
	from modshogun import GaussianBlobsDataGenerator
	from modshogun import GaussianKernel, CombinedKernel
	from modshogun import LinearTimeMMD
	from modshogun import MMDKernelSelectionCombMaxL2
	from modshogun import MMDKernelSelectionCombOpt
	from modshogun import PERMUTATION, MMD1_GAUSSIAN, WITHIN_BLOCK_DIRECT
	from modshogun import EuclideanDistance
	from modshogun import Statistics, Math

	# init seed for reproducability
	Math.init_random(1234)

	# note that the linear time statistic is designed for much larger datasets
	# results for this low number will be bad (unstable, type I error wrong)

	# streaming data generator
	gen_p=GaussianBlobsDataGenerator(num_blobs, distance, 1, 0)
	gen_q=GaussianBlobsDataGenerator(num_blobs, distance, stretch, angle)

	# stream some data and plot
	num_plot=1000
	features=gen_p.get_streamed_features(num_plot)
	features=features.create_merged_copy(gen_q.get_streamed_features(num_plot))
	data=features.get_feature_matrix()

	#figure()
	#subplot(2,2,1)
	#grid(True)
	#plot(data[0][0:num_plot], data[1][0:num_plot], 'r.', label='$x$')
	#title('$X\sim p$')
	#subplot(2,2,2)
	#grid(True)
	#plot(data[0][num_plot+1:2*num_plot], data[1][num_plot+1:2*num_plot], 'b.', label='$x$', alpha=0.5)
	#title('$Y\sim q$')

	# create combined kernel with Gaussian kernels inside (shoguns Gaussian kernel is
	# different to the standard form, see documentation)
	sigmas=[2**x for x in range(-3,10)]
	widths=[x*x*2 for x in sigmas]
	combined=CombinedKernel()
	for i in range(len(sigmas)):
		combined.append_kernel(GaussianKernel(10, widths[i]))

	# mmd instance using streaming features, blocksize of 8
	block_size=8
	mmd=LinearTimeMMD(combined, gen_p, gen_q, m, block_size)
	mmd.set_null_var_est_method(WITHIN_BLOCK_DIRECT)

	# kernel selection instance (this can easily replaced by the other methods for selecting
	# combined kernels
	if selection_method=="opt":
		selection=MMDKernelSelectionCombOpt(mmd)
	elif selection_method=="l2":
		selection=MMDKernelSelectionCombMaxL2(mmd)

	# perform kernel selection (kernel is automatically set)
	kernel=selection.select_kernel()
	kernel=CombinedKernel.obtain_from_generic(kernel)
	#print "selected kernel weights:", kernel.get_subkernel_weights()
	#subplot(2,2,3)
	#plot(kernel.get_subkernel_weights())
	#title("Kernel weights")

	# compute tpye I and II error (use many more trials). Type I error is only
	# estimated to check MMD1_GAUSSIAN method for estimating the null
	# distribution. Note that testing has to happen on difference data than
	# kernel selecting, but the linear time mmd does this implicitly
	mmd.set_null_approximation_method(MMD1_GAUSSIAN)

	# number of trials should be larger to compute tight confidence bounds
	num_trials=5;
	alpha=0.05 # test power
	typeIerrors=[0 for x in range(num_trials)]
	typeIIerrors=[0 for x in range(num_trials)]
	for i in range(num_trials):
		# this effectively means that p=q - rejecting is tpye I error
		mmd.set_simulate_h0(True)
		typeIerrors[i]=mmd.perform_test()>alpha
		mmd.set_simulate_h0(False)

		typeIIerrors[i]=mmd.perform_test()>alpha

	#print "type I error:", mean(typeIerrors), ", type II error:", mean(typeIIerrors)

	return kernel,typeIerrors,typeIIerrors

if __name__=='__main__':
	print('MMDKernelSelectionCombined')
	statistics_mmd_kernel_selection_combined(*parameter_list[0])
	#show()
