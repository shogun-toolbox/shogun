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

from shogun.Features import RealFeatures
from shogun.Features import DataGenerator
from shogun.Kernel import GaussianKernel
from shogun.Statistics import HSIC
from shogun.Statistics import BOOTSTRAP, HSIC_GAMMA

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

# use a kernel width of sigma=2, which is 8 in SHOGUN's parametrization
# which is k(x,y)=exp(-||x-y||^2 / tau), in constrast to the standard
# k(x,y)=exp(-||x-y||^2 / (2*sigma^2)), so tau=2*sigma^2
# Note that kernels per data can be different
kernel_x=GaussianKernel(10,8)
kernel_y=GaussianKernel(10,8)

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
plot(data[0], data[1], 'o')
title('Data, rotation=$\pi$/'+str(1/angle*pi)+'\nm='+str(m))
xlabel('$x$')
ylabel('$y$')
grid(True)

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
hist(alt_samples, 20, normed=True);
axvline(thresh_boot, 0, 1, linewidth=2, color='red')
type_two_error=sum(alt_samples<thresh_boot)/float(num_null_samples)
title('Alternative Dist.\n' + 'Type II error is ' + str(type_two_error))
grid(True)

# compute range for all null distribution histograms
hist_range=[min([min(null_samples_boot), min(null_samples_gamma)]), max([max(null_samples_boot), max(null_samples_gamma)])]

# plot null distribution with threshold
subplot(2,2,3)
hist(null_samples_boot, 20, range=hist_range, normed=True);
axvline(thresh_boot, 0, 1, linewidth=2, color='red')
title('Bootstrapped Null Dist.\n' + 'Type I error is '  + str(type_one_error_boot))
grid(True)

# plot null distribution gamma
subplot(2,2,4)
hist(null_samples_gamma, 20, range=hist_range, normed=True);
axvline(thresh_gamma, 0, 1, linewidth=2, color='red')
title('Null Dist. Gamma\nType I error is '  + str(type_one_error_gamma))
grid(True)

# pull plots a bit apart
subplots_adjust(hspace=0.5)
subplots_adjust(wspace=0.5)
show()
