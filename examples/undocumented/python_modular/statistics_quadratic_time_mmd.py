#!/usr/bin/env python
#
# This program is free software you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation either version 3 of the License, or
# (at your option) any later version.
#
# Written (C) 2013 Heiko Strathmann
#
from numpy import *

def statistics_quadratic_time_mmd ():
	from shogun.Features import RealFeatures
	from shogun.Features import MeanShiftDataGenerator
	from shogun.Kernel import GaussianKernel, CustomKernel
	from shogun.Statistics import QuadraticTimeMMD
	from shogun.Statistics import BOOTSTRAP, MMD2_SPECTRUM, MMD2_GAMMA, BIASED, UNBIASED
	from shogun.Mathematics import Statistics, IntVector, RealVector

	# number of examples kept low in order to make things fast
	m=30;
	dim=2;
	difference=0.5;

	# streaming data generator for mean shift distributions
	gen_p=MeanShiftDataGenerator(0, dim);
	gen_q=MeanShiftDataGenerator(difference, dim);

	# stream some data from generator
	feat_p=gen_p.get_streamed_features(m);
	feat_q=gen_q.get_streamed_features(m);

	# set kernel a-priori. usually one would do some kernel selection. See
	# other examples for this.
	width=10;
	kernel=GaussianKernel(10, width);

	# create quadratic time mmd instance. Note that this constructor
	# copies p and q and does not reference them
	mmd=QuadraticTimeMMD(kernel, feat_p, feat_q);

	# perform test: compute p-value and test if null-hypothesis is rejected for
	# a test level of 0.05
	alpha=0.05;
	
	# using bootstrapping (slow, not the most reliable way. Consider pre-
	# computing the kernel when using it, see below).
	# Also, in practice, use at least 250 iterations
	mmd.set_null_approximation_method(BOOTSTRAP);
	mmd.set_bootstrap_iterations(3);
	p_value=mmd.perform_test();
	# reject if p-value is smaller than test level
	print "bootstrap: p!=q: ", p_value<alpha

	# using spectrum method. Use at least 250 samples from null.
	# This is consistent but sometimes breaks, always monitor type I error.
	# See tutorial for number of eigenvalues to use .
	# Only works with BIASED statistic
	mmd.set_statistic_type(BIASED);
	mmd.set_null_approximation_method(MMD2_SPECTRUM);
	mmd.set_num_eigenvalues_spectrum(3);
	mmd.set_num_samples_sepctrum(250);
	p_value=mmd.perform_test();
	# reject if p-value is smaller than test level
	print "spectrum: p!=q: ", p_value<alpha

	# using gamma method. This is a quick hack, which works most of the time
	# but is NOT guaranteed to. See tutorial for details.
	# Only works with BIASED statistic
	mmd.set_statistic_type(BIASED);
	mmd.set_null_approximation_method(MMD2_GAMMA);
	p_value=mmd.perform_test();
	# reject if p-value is smaller than test level
	print "gamma: p!=q: ", p_value<alpha

	# compute tpye I and II error (use many more trials in practice).
	# Type I error is not necessary if one uses bootstrapping. We do it here
	# anyway, but note that this is an efficient way of computing it.
	# Also note that testing has to happen on
	# difference data than kernel selection, but the linear time mmd does this
	# implicitly and we used a fixed kernel here.
	mmd.set_null_approximation_method(BOOTSTRAP);
	mmd.set_bootstrap_iterations(5);
	num_trials=5;
	type_I_errors=RealVector(num_trials);
	type_II_errors=RealVector(num_trials);
	inds=int32(array([x for x in range(2*m)])) # numpy
	p_and_q=mmd.get_p_and_q();

	# use a precomputed kernel to be faster
	kernel.init(p_and_q, p_and_q);
	precomputed=CustomKernel(kernel);
	mmd.set_kernel(precomputed);
	for i in range(num_trials):
		# this effectively means that p=q - rejecting is tpye I error
		inds=random.permutation(inds) # numpy permutation
		precomputed.add_row_subset(inds);
		precomputed.add_col_subset(inds);
		type_I_errors[i]=mmd.perform_test()>alpha;
		precomputed.remove_row_subset();
		precomputed.remove_col_subset();

		# on normal data, this gives type II error
		type_II_errors[i]=mmd.perform_test()>alpha;
	
if __name__=='__main__':
	print('QuadraticTimeMMD')
	statistics_quadratic_time_mmd()
