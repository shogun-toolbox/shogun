#
# Copyright (c) The Shogun Machine Learning Toolbox
# Written (w) 2012-2013 Heiko Strathmann
# Written (w) 2014 Soumyajit De
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# The views and conclusions contained in the software and documentation are those
# of the authors and should not be interpreted as representing official policies,
# either expressed or implied, of the Shogun Development Team.
#

import numpy as np
import pylab as plt
import scipy.stats as stats

from modshogun import RealFeatures
from modshogun import MeanShiftDataGenerator
from modshogun import GaussianKernel
from modshogun import BTestMMD
from modshogun import PERMUTATION, MMD1_GAUSSIAN
from modshogun import Statistics, Math

# for nice plotting that fits into our shogun tutorial
#import latex_plot_inits


# The following test recreates the plots at figure (1) from [1].
#
# [1] W. Zaremba, A. Gretton, and M. Blaschko, B-test: A Non-parametric,
# Low Variance Kernel Two-sample Test. In Advances in Neural Information
# Processing Systems (NIPS), 2013.
def btest_mmd_graphical():

	# parameters, change to get different results
	m=250 # set to 10000 for a good test result
	dim=2

	# setting the difference of the first dimension. smaller makes a harder test
	difference=0.5

	# number of samples taken from null and alternative distribution
	num_null_samples=250

	# streaming data generator for mean shift distributions
	gen_p=MeanShiftDataGenerator(0, dim)
	gen_q=MeanShiftDataGenerator(difference, dim)

	# use a gaussian kernel of fixed width. usually kernel selection is performed
	# to adjust the width
	sigma=2
	width=sigma*sigma*2
	kernel=GaussianKernel(10, width)

	# mmd instance using streaming features
	mmd=BTestMMD(kernel, gen_p, gen_q, m)

	# using smaller blocksize (4), 2 samples from p and 2 samples from q in each block
	mmd.set_blocksize(4)

	# sample alternative distribution, stream ensures different samples each run
	alt_samples_small=np.zeros(num_null_samples)
	for i in range(len(alt_samples_small)):
		alt_samples_small[i]=mmd.compute_statistic()

	# fit a normal distribution to alternative
	alt_variance_small=np.var(alt_samples_small)

	# sample from null distribution
	# bootstrapping, biased statistic
	mmd.set_null_approximation_method(PERMUTATION)
	mmd.set_num_null_samples(num_null_samples)
	null_samples_small=mmd.sample_null()

	# fit normal distribution to null and sample a normal distribution
	mmd.set_null_approximation_method(MMD1_GAUSSIAN)
	null_variance_small=mmd.compute_variance_estimate()
	#null_samples_small_gaussian=normal(0,sqrt(variance),num_null_samples_small)

	# using larger blocksize (250), 125 samples from p and 125 samples from q in each block
	mmd.set_blocksize(250)

	# sample alternative distribution, stream ensures different samples each run
	alt_samples_large=np.zeros(num_null_samples)
	for i in range(len(alt_samples_large)):
		alt_samples_large[i]=mmd.compute_statistic()

	# fit a normal distribution to alternative
	alt_variance_large=np.var(alt_samples_large)

	# sample from null distribution
	# bootstrapping, biased statistic
	mmd.set_null_approximation_method(PERMUTATION)
	mmd.set_num_null_samples(num_null_samples)
	null_samples_large=mmd.sample_null()

	# fit normal distribution to null and sample a normal distribution
	mmd.set_null_approximation_method(MMD1_GAUSSIAN)
	null_variance_large=mmd.compute_variance_estimate()

	# to plot data, sample a few examples from stream first
	features=gen_p.get_streamed_features(m)
	features=features.create_merged_copy(gen_q.get_streamed_features(m))
	data=features.get_feature_matrix()

	# plot
	plt.figure()

	# plot data of p and q
	plt.subplot(2,4,1)
	plt.grid(True)
	plt.gca().xaxis.set_major_locator( plt.MaxNLocator(nbins = 4) ) # reduce number of x-ticks
	plt.gca().yaxis.set_major_locator( plt.MaxNLocator(nbins = 4) ) # reduce number of x-ticks
	plt.plot(data[0][0:m], data[1][0:m], 'ro', label='$x$')
	plt.plot(data[0][m+1:2*m], data[1][m+1:2*m], 'bo', label='$x$', alpha=0.5)
	plt.title('Data, shift in $x_1$='+str(difference)+'\nm='+str(m))
	plt.xlabel('$x_1, y_1$')
	plt.ylabel('$x_2, y_2$')

	# histogram of first data dimension and pdf
	plt.subplot(2,4,5)
	plt.grid(True)
	plt.gca().xaxis.set_major_locator( plt.MaxNLocator(nbins = 3) ) # reduce number of x-ticks
	plt.gca().yaxis.set_major_locator( plt.MaxNLocator(nbins = 3) ) # reduce number of x-ticks
	plt.hist(data[0], bins=50, alpha=0.5, facecolor='r', normed=True)
	plt.hist(data[1], bins=50, alpha=0.5, facecolor='b', normed=True)
	xs=np.linspace(min(data[0])-1,max(data[0])+1, 50)
	plt.plot(xs,plt.normpdf( xs, 0, 1), 'r', linewidth=3)
	plt.plot(xs,plt.normpdf( xs, difference, 1), 'b', linewidth=3)
	plt.xlabel('$x_1, y_1$')
	plt.ylabel('$p(x_1), p(y_1)$')
	plt.title('Data PDF in $x_1, y_1$')

	# compute threshold for test level
	alpha=0.05
	null_samples_small.sort()
	thresh=null_samples_small[np.floor(len(null_samples_small)*(1-alpha))];

	# plot null and alternative distribution with normal approximation and thereshold
	plt.subplot(2,4,2)
	plt.grid(True)
	plt.gca().xaxis.set_major_locator(plt.MaxNLocator(nbins = 3)) # reduce number of x-ticks
	plt.gca().yaxis.set_major_locator(plt.MaxNLocator(nbins = 3)) # reduce number of y-ticks

	# plot alternative distribution with threshold
	plt.hist(alt_samples_small, 30, normed=True, color='red', label='$\mathcal{H}_1$')
	plt.axvline(thresh, 0, 1, linewidth=2, color='blue')

	# plot a gaussian approximation for alternative
	xs = np.linspace(min(alt_samples_small)-1, max(alt_samples_small)+1, 50)
	plt.plot(xs, plt.normpdf(xs, np.mean(alt_samples_small), np.sqrt(alt_variance_small)), color='blue')

	# plot null distribution and gaussian approximation for null
	plt.hist(null_samples_small, 30, normed=True, color='green', label='$\mathcal{H}_0$')

	xs = np.linspace(min(null_samples_small)-1, max(null_samples_small)+1, 50)
	plt.plot(xs, plt.normpdf(xs, 0, np.sqrt(null_variance_small)), color='black')
	plt.legend(loc='upper right')
	plt.ylabel('Density')
	plt.title('Distribution under $\mathcal{H}_0$ and $\mathcal{H}_1$\n for $B=4$')

	# Q-Q plot for both null and alternative
	plt.subplot(2,4,3)
	plt.grid(True)
	plt.gca().xaxis.set_major_locator(plt.MaxNLocator(nbins = 3)) # reduce number of x-ticks
	plt.gca().yaxis.set_major_locator(plt.MaxNLocator(nbins = 3)) # reduce number of y-ticks
	stats.probplot(alt_samples_small, dist="norm", plot=plt)
	plt.title('Q-Q plot for $\mathcal{H}_1$')

	plt.subplot(2,4,4)
	plt.grid(True)
	plt.gca().xaxis.set_major_locator(plt.MaxNLocator(nbins = 3)) # reduce number of x-ticks
	plt.gca().yaxis.set_major_locator(plt.MaxNLocator(nbins = 3)) # reduce number of y-ticks
	stats.probplot(null_samples_small, dist="norm", plot=plt)
	plt.title('Q-Q plot for $\mathcal{H}_0$')

	# compute threshold for test level
	alpha=0.05
	null_samples_large.sort()
	thresh=null_samples_large[np.floor(len(null_samples_large)*(1-alpha))];

	# plot null and alternative distribution with normal approximation and thereshold
	plt.subplot(2,4,6)
	plt.grid(True)
	plt.gca().xaxis.set_major_locator(plt.MaxNLocator(nbins = 3)) # reduce number of x-ticks
	plt.gca().yaxis.set_major_locator(plt.MaxNLocator(nbins = 3)) # reduce number of y-ticks

	# plot alternative distribution with threshold
	plt.hist(alt_samples_large, 30, normed=True, color='red', label='$\mathcal{H}_1$')
	plt.axvline(thresh, 0, 1, linewidth=2, color='blue')

	# plot a gaussian approximation for alternative
	xs = np.linspace(min(alt_samples_large)-1, max(alt_samples_large)+1, 50)
	plt.plot(xs, plt.normpdf(xs, np.mean(alt_samples_large), np.sqrt(alt_variance_large)), color='blue')

	# plot null distribution and gaussian approximation for null
	plt.hist(null_samples_large, 30, normed=True, color='green', label='$\mathcal{H}_0$')

	xs = np.linspace(min(null_samples_large)-1, max(null_samples_large)+1, 50)
	plt.plot(xs, plt.normpdf(xs, 0, np.sqrt(null_variance_large)), color='black')
	plt.legend(loc='upper right')
	plt.ylabel('Density')
	plt.title('Distribution under $\mathcal{H}_0$ and $\mathcal{H}_1$\n for $B=250$')

	# Q-Q plot for both null and alternative
	plt.subplot(2,4,7)
	plt.grid(True)
	plt.gca().xaxis.set_major_locator(plt.MaxNLocator(nbins = 3)) # reduce number of x-ticks
	plt.gca().yaxis.set_major_locator(plt.MaxNLocator(nbins = 3)) # reduce number of y-ticks
	stats.probplot(alt_samples_large, dist="norm", plot=plt)
	plt.title('Q-Q plot for $\mathcal{H}_1$')

	plt.subplot(2,4,8)
	plt.grid(True)
	plt.gca().xaxis.set_major_locator(plt.MaxNLocator(nbins = 3)) # reduce number of x-ticks
	plt.gca().yaxis.set_major_locator(plt.MaxNLocator(nbins = 3)) # reduce number of y-ticks
	stats.probplot(null_samples_large, dist="norm", plot=plt)
	plt.title('Q-Q plot for $\mathcal{H}_0$')

	# pull plots a bit apart
	plt.subplots_adjust(hspace=0.3)
	plt.subplots_adjust(wspace=0.3)

if __name__=='__main__':
	btest_mmd_graphical()
	plt.show()
