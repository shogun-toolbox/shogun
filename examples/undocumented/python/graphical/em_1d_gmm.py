from pylab import figure,show,connect,hist,plot,legend
from numpy import array, append, arange, empty, exp
from modshogun import Gaussian, GMM
from modshogun import RealFeatures
import util

util.set_title('EM for 1d GMM example')

#set the parameters
min_cov=1e-9
max_iter=1000
min_change=1e-9

#setup the real GMM
real_gmm=GMM(3)

real_gmm.set_nth_mean(array([-2.0]), 0)
real_gmm.set_nth_mean(array([0.0]), 1)
real_gmm.set_nth_mean(array([2.0]), 2)

real_gmm.set_nth_cov(array([[0.3]]), 0)
real_gmm.set_nth_cov(array([[0.1]]), 1)
real_gmm.set_nth_cov(array([[0.2]]), 2)

real_gmm.set_coef(array([0.3, 0.5, 0.2]))

#generate training set from real GMM
generated=array([real_gmm.sample()])
for i in range(199):
    generated=append(generated, array([real_gmm.sample()]), axis=1)

feat_train=RealFeatures(generated)

#train GMM using EM
est_gmm=GMM(3)
est_gmm.train(feat_train)
est_gmm.train_em(min_cov, max_iter, min_change)

#get and print estimated means and covariances
est_mean1=est_gmm.get_nth_mean(0)
est_mean2=est_gmm.get_nth_mean(1)
est_mean3=est_gmm.get_nth_mean(2)
est_cov1=est_gmm.get_nth_cov(0)
est_cov2=est_gmm.get_nth_cov(1)
est_cov3=est_gmm.get_nth_cov(2)
est_coef=est_gmm.get_coef()
print est_mean1
print est_cov1
print est_mean2
print est_cov2
print est_mean3
print est_cov3
print est_coef

#plot real GMM, data and estimated GMM
min_gen=min(min(generated))
max_gen=max(max(generated))
plot_real=empty(0)
plot_est=empty(0)
for i in arange(min_gen, max_gen, 0.001):
    plot_real=append(plot_real, array([exp(real_gmm.cluster(array([i]))[3])]))
    plot_est=append(plot_est, array([exp(est_gmm.cluster(array([i]))[3])]))
real_plot=plot(arange(min_gen, max_gen, 0.001), plot_real, "b")
est_plot=plot(arange(min_gen, max_gen, 0.001), plot_est, "r")
real_hist=hist(generated.transpose(), bins=50, normed=True, fc="gray")
legend(("Real GMM", "Estimated GMM"))
connect('key_press_event', util.quit)
show()
