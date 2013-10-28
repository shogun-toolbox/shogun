#
# This program is free software you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation either version 3 of the License, or
# (at your option) any later version.
#
# Written (C) 2012 Heiko Strathmann
#
from numpy import *
from pylab import *
from scipy import *

from modshogun import RealFeatures
from modshogun import DataGenerator
from modshogun import GaussianKernel
from modshogun import HSIC
from modshogun import BOOTSTRAP, HSIC_GAMMA
from modshogun import EuclideanDistance
from modshogun import Statistics, Math

# for nice plotting that fits into our shogun tutorial
import latex_plot_inits

def hsic_graphical():
	# parameters, change to get different results
	m=250
	difference=3

	# setting the angle lower makes a harder test
	angle=pi/30

	# number of samples taken from null and alternative distribution
	num_null_samples=500

	# use data generator class to produce example data
	data=DataGenerator.generate_sym_mix_gauss(m,difference,angle)

	# create shogun feature representation
	features_x=RealFeatures(array([data[0]]))
	features_y=RealFeatures(array([data[1]]))

	# compute median data distance in order to use for Gaussian kernel width
	# 0.5*median_distance normally (factor two in Gaussian kernel)
	# However, shoguns kernel width is different to usual parametrization
	# Therefore 0.5*2*median_distance^2
	# Use a subset of data for that, only 200 elements. Median is stable
	subset=int32(array([x for x in range(features_x.get_num_vectors())])) # numpy
	subset=random.permutation(subset) # numpy permutation
	subset=subset[0:200]
	features_x.add_subset(subset)
	dist=EuclideanDistance(features_x, features_x)
	distances=dist.get_distance_matrix()
	features_x.remove_subset()
	median_distance=Statistics.matrix_median(distances, True)
	sigma_x=median_distance**2
	features_y.add_subset(subset)
	dist=EuclideanDistance(features_y, features_y)
	distances=dist.get_distance_matrix()
	features_y.remove_subset()
	median_distance=Statistics.matrix_median(distances, True)
	sigma_y=median_distance**2
	print "median distance for Gaussian kernel on x:", sigma_x
	print "median distance for Gaussian kernel on y:", sigma_y
	kernel_x=GaussianKernel(10,sigma_x)
	kernel_y=GaussianKernel(10,sigma_y)

	# create hsic instance. Note that this is a convienience constructor which copies
	# feature data. features_x and features_y are not these used in hsic.
	# This is only for user-friendlyness. Usually, its ok to do this.
	# Below, the alternative distribution is sampled, which means
	# that new feature objects have to be created in each iteration (slow)
	# However, normally, the alternative distribution is not sampled
	hsic=HSIC(kernel_x,kernel_y,features_x,features_y)

	# sample alternative distribution
	alt_samples=zeros(num_null_samples)
	for i in range(len(alt_samples)):
		data=DataGenerator.generate_sym_mix_gauss(m,difference,angle)
		features_x.set_feature_matrix(array([data[0]]))
		features_y.set_feature_matrix(array([data[1]]))

		# re-create hsic instance everytime since feature objects are copied due to
		# useage of convienience constructor
		hsic=HSIC(kernel_x,kernel_y,features_x,features_y)
		alt_samples[i]=hsic.compute_statistic()

	# sample from null distribution
	# bootstrapping, biased statistic
	hsic.set_null_approximation_method(BOOTSTRAP)
	hsic.set_bootstrap_iterations(num_null_samples)
	null_samples_boot=hsic.bootstrap_null()

	# fit gamma distribution, biased statistic
	hsic.set_null_approximation_method(HSIC_GAMMA)
	gamma_params=hsic.fit_null_gamma()
	# sample gamma with parameters
	null_samples_gamma=array([gamma(gamma_params[0], gamma_params[1]) for _ in range(num_null_samples)])

	# plot
	figure()

	# plot data x and y
	subplot(2,2,1)
	gca().xaxis.set_major_locator( MaxNLocator(nbins = 4) ) # reduce number of x-ticks
	gca().yaxis.set_major_locator( MaxNLocator(nbins = 4) ) # reduce number of x-ticks
	grid(True)
	plot(data[0], data[1], 'o')
	title('Data, rotation=$\pi$/'+str(1/angle*pi)+'\nm='+str(m))
	xlabel('$x$')
	ylabel('$y$')

	# compute threshold for test level
	alpha=0.05
	null_samples_boot.sort()
	null_samples_gamma.sort()
	thresh_boot=null_samples_boot[floor(len(null_samples_boot)*(1-alpha))];
	thresh_gamma=null_samples_gamma[floor(len(null_samples_gamma)*(1-alpha))];

	type_one_error_boot=sum(null_samples_boot<thresh_boot)/float(num_null_samples)
	type_one_error_gamma=sum(null_samples_gamma<thresh_boot)/float(num_null_samples)

	# plot alternative distribution with threshold
	subplot(2,2,2)
	gca().xaxis.set_major_locator( MaxNLocator(nbins = 3) ) # reduce number of x-ticks
	gca().yaxis.set_major_locator( MaxNLocator(nbins = 3) ) # reduce number of x-ticks
	grid(True)
	hist(alt_samples, 20, normed=True);
	axvline(thresh_boot, 0, 1, linewidth=2, color='red')
	type_two_error=sum(alt_samples<thresh_boot)/float(num_null_samples)
	title('Alternative Dist.\n' + 'Type II error is ' + str(type_two_error))

	# compute range for all null distribution histograms
	hist_range=[min([min(null_samples_boot), min(null_samples_gamma)]), max([max(null_samples_boot), max(null_samples_gamma)])]

	# plot null distribution with threshold
	subplot(2,2,3)
	gca().xaxis.set_major_locator( MaxNLocator(nbins = 3) ) # reduce number of x-ticks
	gca().yaxis.set_major_locator( MaxNLocator(nbins = 3) ) # reduce number of x-ticks
	grid(True)
	hist(null_samples_boot, 20, range=hist_range, normed=True);
	axvline(thresh_boot, 0, 1, linewidth=2, color='red')
	title('Bootstrapped Null Dist.\n' + 'Type I error is '  + str(type_one_error_boot))

	# plot null distribution gamma
	subplot(2,2,4)
	gca().xaxis.set_major_locator( MaxNLocator(nbins = 3) ) # reduce number of x-ticks
	gca().yaxis.set_major_locator( MaxNLocator(nbins = 3) ) # reduce number of x-ticks
	grid(True)
	hist(null_samples_gamma, 20, range=hist_range, normed=True);
	axvline(thresh_gamma, 0, 1, linewidth=2, color='red')
	title('Null Dist. Gamma\nType I error is '  + str(type_one_error_gamma))
	grid(True)

	# pull plots a bit apart
	subplots_adjust(hspace=0.5)
	subplots_adjust(wspace=0.5)

if __name__=='__main__':
	hsic_graphical()
	show()
