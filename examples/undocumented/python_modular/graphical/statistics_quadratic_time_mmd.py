#
# This program is free software you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation either version 3 of the License, or
# (at your option) any later version.
#
# Written (C) 2012-2013 Heiko Strathmann
#
from numpy import *
from pylab import *
from scipy import *

from modshogun import RealFeatures
from modshogun import MeanShiftDataGenerator
from modshogun import GaussianKernel, CombinedKernel
from modshogun import QuadraticTimeMMD, MMDKernelSelectionMax
from modshogun import PERMUTATION, MMD2_SPECTRUM, MMD2_GAMMA, BIASED, UNBIASED
from modshogun import EuclideanDistance
from modshogun import Statistics, Math

# for nice plotting that fits into our shogun tutorial
import latex_plot_inits

def quadratic_time_mmd_graphical():

	# parameters, change to get different results
	m=100
	dim=2

	# setting the difference of the first dimension smaller makes a harder test
	difference=0.5

	# number of samples taken from null and alternative distribution
	num_null_samples=500

	# streaming data generator for mean shift distributions
	gen_p=MeanShiftDataGenerator(0, dim)
	gen_q=MeanShiftDataGenerator(difference, dim)

	# Stream examples and merge them in order to compute MMD on joint sample
	# alternative is to call a different constructor of QuadraticTimeMMD
	features=gen_p.get_streamed_features(m)
	features=features.create_merged_copy(gen_q.get_streamed_features(m))

	# use the median kernel selection
	# create combined kernel with Gaussian kernels inside (shoguns Gaussian kernel is
	# compute median data distance in order to use for Gaussian kernel width
	# 0.5*median_distance normally (factor two in Gaussian kernel)
	# However, shoguns kernel width is different to usual parametrization
	# Therefore 0.5*2*median_distance^2
	# Use a subset of data for that, only 200 elements. Median is stable
	sigmas=[2**x for x in range(-3,10)]
	widths=[x*x*2 for x in sigmas]
	print "kernel widths:", widths
	combined=CombinedKernel()
	for i in range(len(sigmas)):
		combined.append_kernel(GaussianKernel(10, widths[i]))

	# create MMD instance, use biased statistic
	mmd=QuadraticTimeMMD(combined,features, m)
	mmd.set_statistic_type(BIASED)

	# kernel selection instance (this can easily replaced by the other methods for selecting
	# single kernels
	selection=MMDKernelSelectionMax(mmd)

	# perform kernel selection
	kernel=selection.select_kernel()
	kernel=GaussianKernel.obtain_from_generic(kernel)
	mmd.set_kernel(kernel);
	print "selected kernel width:", kernel.get_width()

	# sample alternative distribution (new data each trial)
	alt_samples=zeros(num_null_samples)
	for i in range(len(alt_samples)):
		# Stream examples and merge them in order to replace in MMD
		features=gen_p.get_streamed_features(m)
		features=features.create_merged_copy(gen_q.get_streamed_features(m))
		mmd.set_p_and_q(features)
		alt_samples[i]=mmd.compute_statistic()

	# sample from null distribution
	# bootstrapping, biased statistic
	mmd.set_null_approximation_method(PERMUTATION)
	mmd.set_statistic_type(BIASED)
	mmd.set_num_null_samples(num_null_samples)
	null_samples_boot=mmd.sample_null()

	# sample from null distribution
	# spectrum, biased statistic
	if "sample_null_spectrum" in dir(QuadraticTimeMMD):
			mmd.set_null_approximation_method(MMD2_SPECTRUM)
			mmd.set_statistic_type(BIASED)
			null_samples_spectrum=mmd.sample_null_spectrum(num_null_samples, m-10)

	# fit gamma distribution, biased statistic
	mmd.set_null_approximation_method(MMD2_GAMMA)
	mmd.set_statistic_type(BIASED)
	gamma_params=mmd.fit_null_gamma()
	# sample gamma with parameters
	null_samples_gamma=array([gamma(gamma_params[0], gamma_params[1]) for _ in range(num_null_samples)])

	# to plot data, sample a few examples from stream first
	features=gen_p.get_streamed_features(m)
	features=features.create_merged_copy(gen_q.get_streamed_features(m))
	data=features.get_feature_matrix()

	# plot
	figure()
	title('Quadratic Time MMD')

	# plot data of p and q
	subplot(2,3,1)
	grid(True)
	gca().xaxis.set_major_locator( MaxNLocator(nbins = 4) ) # reduce number of x-ticks
	gca().yaxis.set_major_locator( MaxNLocator(nbins = 4) ) # reduce number of x-ticks
	plot(data[0][0:m], data[1][0:m], 'ro', label='$x$')
	plot(data[0][m+1:2*m], data[1][m+1:2*m], 'bo', label='$x$', alpha=0.5)
	title('Data, shift in $x_1$='+str(difference)+'\nm='+str(m))
	xlabel('$x_1, y_1$')
	ylabel('$x_2, y_2$')

	# histogram of first data dimension and pdf
	subplot(2,3,2)
	grid(True)
	gca().xaxis.set_major_locator( MaxNLocator(nbins = 3) ) # reduce number of x-ticks
	gca().yaxis.set_major_locator( MaxNLocator(nbins = 3 )) # reduce number of x-ticks
	hist(data[0], bins=50, alpha=0.5, facecolor='r', normed=True)
	hist(data[1], bins=50, alpha=0.5, facecolor='b', normed=True)
	xs=linspace(min(data[0])-1,max(data[0])+1, 50)
	plot(xs,normpdf( xs, 0, 1), 'r', linewidth=3)
	plot(xs,normpdf( xs, difference, 1), 'b', linewidth=3)
	xlabel('$x_1, y_1$')
	ylabel('$p(x_1), p(y_1)$')
	title('Data PDF in $x_1, y_1$')

	# compute threshold for test level
	alpha=0.05
	null_samples_boot.sort()
	null_samples_spectrum.sort()
	null_samples_gamma.sort()
	thresh_boot=null_samples_boot[floor(len(null_samples_boot)*(1-alpha))];
	thresh_spectrum=null_samples_spectrum[floor(len(null_samples_spectrum)*(1-alpha))];
	thresh_gamma=null_samples_gamma[floor(len(null_samples_gamma)*(1-alpha))];

	type_one_error_boot=sum(null_samples_boot<thresh_boot)/float(num_null_samples)
	type_one_error_spectrum=sum(null_samples_spectrum<thresh_boot)/float(num_null_samples)
	type_one_error_gamma=sum(null_samples_gamma<thresh_boot)/float(num_null_samples)

	# plot alternative distribution with threshold
	subplot(2,3,4)
	grid(True)
	gca().xaxis.set_major_locator( MaxNLocator(nbins = 3) ) # reduce number of x-ticks
	gca().yaxis.set_major_locator( MaxNLocator(nbins = 3) ) # reduce number of x-ticks
	hist(alt_samples, 20, normed=True);
	axvline(thresh_boot, 0, 1, linewidth=2, color='red')
	type_two_error=sum(alt_samples<thresh_boot)/float(num_null_samples)
	title('Alternative Dist.\n' + 'Type II error is ' + str(type_two_error))

	# compute range for all null distribution histograms
	hist_range=[min([min(null_samples_boot), min(null_samples_spectrum), min(null_samples_gamma)]), max([max(null_samples_boot), max(null_samples_spectrum), max(null_samples_gamma)])]

	# plot null distribution with threshold
	subplot(2,3,3)
	gca().xaxis.set_major_locator( MaxNLocator(nbins = 3) ) # reduce number of x-ticks
	gca().yaxis.set_major_locator( MaxNLocator(nbins = 3 )) # reduce number of x-ticks
	hist(null_samples_boot, 20, range=hist_range, normed=True);
	axvline(thresh_boot, 0, 1, linewidth=2, color='red')
	title('Sampled Null Dist.\n' + 'Type I error is '  + str(type_one_error_boot))
	grid(True)

	# plot null distribution spectrum
	subplot(2,3,5)
	grid(True)
	gca().xaxis.set_major_locator( MaxNLocator(nbins = 3) ) # reduce number of x-ticks
	gca().yaxis.set_major_locator( MaxNLocator(nbins = 3) ) # reduce number of x-ticks
	hist(null_samples_spectrum, 20, range=hist_range, normed=True);
	axvline(thresh_spectrum, 0, 1, linewidth=2, color='red')
	title('Null Dist. Spectrum\nType I error is '  + str(type_one_error_spectrum))

	# plot null distribution gamma
	subplot(2,3,6)
	grid(True)
	gca().xaxis.set_major_locator( MaxNLocator(nbins = 3) ) # reduce number of x-ticks
	gca().yaxis.set_major_locator( MaxNLocator(nbins = 3) ) # reduce number of x-ticks
	hist(null_samples_gamma, 20, range=hist_range, normed=True);
	axvline(thresh_gamma, 0, 1, linewidth=2, color='red')
	title('Null Dist. Gamma\nType I error is '  + str(type_one_error_gamma))

	# pull plots a bit apart
	subplots_adjust(hspace=0.5)
	subplots_adjust(wspace=0.5)

if __name__=='__main__':
	quadratic_time_mmd_graphical()
	show()
