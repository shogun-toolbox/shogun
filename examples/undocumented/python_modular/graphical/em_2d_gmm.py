from pylab import figure,scatter,contour,show,legend,connect
from numpy import array, append, arange, reshape, empty, exp
from modshogun import Gaussian, GMM
from modshogun import RealFeatures
import util

util.set_title('EM for 2d GMM example')

#set the parameters
min_cov=1e-9
max_iter=1000
min_change=1e-9
cov_type=0

#setup the real GMM
real_gmm=GMM(2)

real_gmm.set_nth_mean(array([1.0, 1.0]), 0)
real_gmm.set_nth_mean(array([-1.0, -1.0]), 1)

real_gmm.set_nth_cov(array([[1.0, 0.2],[0.2, 0.1]]), 0)
real_gmm.set_nth_cov(array([[0.3, 0.1],[0.1, 1.0]]), 1)

real_gmm.set_coef(array([0.3, 0.7]))

#generate training set from real GMM
generated=array([real_gmm.sample()])
for i in range(199):
    generated=append(generated, array([real_gmm.sample()]), axis=0)

generated=generated.transpose()
feat_train=RealFeatures(generated)

#train GMM using EM
est_gmm=GMM(2, cov_type)
est_gmm.train(feat_train)
est_gmm.train_em(min_cov, max_iter, min_change)

#get and print estimated means and covariances
est_mean1=est_gmm.get_nth_mean(0)
est_mean2=est_gmm.get_nth_mean(1)
est_cov1=est_gmm.get_nth_cov(0)
est_cov2=est_gmm.get_nth_cov(1)
est_coef=est_gmm.get_coef()
print est_mean1
print est_cov1
print est_mean2
print est_cov2
print est_coef

#plot real GMM, data and estimated GMM
min_x_gen=min(min(generated[[0]]))-0.1
max_x_gen=max(max(generated[[0]]))+0.1
min_y_gen=min(min(generated[[1]]))-0.1
max_y_gen=max(max(generated[[1]]))+0.1

plot_real=empty(0)
plot_est=empty(0)

for i in arange(min_x_gen, max_x_gen, 0.05):
    for j in arange(min_y_gen, max_y_gen, 0.05):
        plot_real=append(plot_real, array([exp(real_gmm.cluster(array([i, j]))[2])]))
        plot_est=append(plot_est, array([exp(est_gmm.cluster(array([i, j]))[2])]))

plot_real=reshape(plot_real, (arange(min_x_gen, max_x_gen, 0.05).shape[0], arange(min_y_gen, max_y_gen, 0.05).shape[0]))
plot_est=reshape(plot_est, (arange(min_x_gen, max_x_gen, 0.05).shape[0], arange(min_y_gen, max_y_gen, 0.05).shape[0]))

real_plot=contour(arange(min_x_gen, max_x_gen, 0.05), arange(min_y_gen, max_y_gen, 0.05), plot_real.transpose(), colors="b")
est_plot=contour(arange(min_x_gen, max_x_gen, 0.05), arange(min_y_gen, max_y_gen, 0.05), plot_est.transpose(), colors="r")
real_scatter=scatter(generated[[0]], generated[[1]], c="gray")
legend((real_plot.collections[0], est_plot.collections[0]), ("Real GMM", "Estimated GMM"))

connect('key_press_event', util.quit)
show()
