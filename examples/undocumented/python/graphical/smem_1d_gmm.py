from pylab import figure,show,connect,hist,plot,legend
from numpy import array, append, arange, empty, exp
from modshogun import Gaussian, GMM
from modshogun import RealFeatures
import util

util.set_title('SMEM for 1d GMM example')

#set the parameters
max_iter=100
max_cand=5
min_cov=1e-9
max_em_iter=1000
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

#train GMM using SMEM and print log-likelihood
est_smem_gmm=GMM(3)
est_smem_gmm.train(feat_train)
print est_smem_gmm.train_smem(max_iter, max_cand, min_cov, max_em_iter, min_change)
#train GMM using EM and print log-likelihood
est_em_gmm=GMM(3)
est_em_gmm.train(feat_train)
print est_em_gmm.train_em(min_cov, max_em_iter, min_change)

#plot real GMM, data and both estimated GMMs
min_gen=min(min(generated))
max_gen=max(max(generated))
plot_real=empty(0)
plot_est_smem=empty(0)
plot_est_em=empty(0)
for i in arange(min_gen, max_gen, 0.001):
    plot_real=append(plot_real, array([exp(real_gmm.cluster(array([i]))[3])]))
    plot_est_smem=append(plot_est_smem, array([exp(est_smem_gmm.cluster(array([i]))[3])]))
    plot_est_em=append(plot_est_em, array([exp(est_em_gmm.cluster(array([i]))[3])]))
real_plot=plot(arange(min_gen, max_gen, 0.001), plot_real, "b")
est_em_plot=plot(arange(min_gen, max_gen, 0.001), plot_est_em, "g")
est_smem_plot=plot(arange(min_gen, max_gen, 0.001), plot_est_smem, "r")
real_hist=hist(generated.transpose(), bins=50, normed=True, fc="gray")
legend(("Real GMM", "Estimated EM GMM", "Estimated SMEM GMM"))
connect('key_press_event', util.quit)
show()
