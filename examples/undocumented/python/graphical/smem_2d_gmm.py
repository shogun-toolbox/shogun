from pylab import figure,scatter,contour,show,legend,connect
from numpy import array, append, arange, reshape, empty, exp
from modshogun import Gaussian, GMM
from modshogun import RealFeatures
import util

util.set_title('SMEM for 2d GMM example')

#set the parameters
max_iter=100
max_cand=5
min_cov=1e-9
max_em_iter=1000
min_change=1e-9
cov_type=0

#setup the real GMM
real_gmm=GMM(3)

real_gmm.set_nth_mean(array([2.0, 2.0]), 0)
real_gmm.set_nth_mean(array([-2.0, -2.0]), 1)
real_gmm.set_nth_mean(array([2.0, -2.0]), 2)

real_gmm.set_nth_cov(array([[1.0, 0.2],[0.2, 0.5]]), 0)
real_gmm.set_nth_cov(array([[0.2, 0.1],[0.1, 0.5]]), 1)
real_gmm.set_nth_cov(array([[0.3, -0.2],[-0.2, 0.8]]), 2)

real_gmm.set_coef(array([0.3, 0.4, 0.3]))

#generate training set from real GMM
generated=array([real_gmm.sample()])
for i in range(199):
    generated=append(generated, array([real_gmm.sample()]), axis=0)

generated=generated.transpose()
feat_train=RealFeatures(generated)

#train GMM using SMEM and print log-likelihood
est_smem_gmm=GMM(3, cov_type)
est_smem_gmm.train(feat_train)

est_smem_gmm.set_nth_mean(array([2.0, 0.0]), 0)
est_smem_gmm.set_nth_mean(array([-2.0, -2.0]), 1)
est_smem_gmm.set_nth_mean(array([-3.0, -3.0]), 2)

est_smem_gmm.set_nth_cov(array([[1.0, 0.0],[0.0, 1.0]]), 0)
est_smem_gmm.set_nth_cov(array([[1.0, 0.0],[0.0, 1.0]]), 1)
est_smem_gmm.set_nth_cov(array([[1.0, 0.0],[0.0, 1.0]]), 2)

est_smem_gmm.set_coef(array([0.3333, 0.3333, 0.3334]))

print est_smem_gmm.train_smem(max_iter, max_cand, min_cov, max_em_iter, min_change)

#train GMM using EM and bad initial conditions and print log-likelihood
est_em_gmm=GMM(3, cov_type)
est_em_gmm.train(feat_train)

est_em_gmm.set_nth_mean(array([2.0, 0.0]), 0)
est_em_gmm.set_nth_mean(array([-2.0, -2.0]), 1)
est_em_gmm.set_nth_mean(array([-3.0, -3.0]), 2)

est_em_gmm.set_nth_cov(array([[1.0, 0.0],[0.0, 1.0]]), 0)
est_em_gmm.set_nth_cov(array([[1.0, 0.0],[0.0, 1.0]]), 1)
est_em_gmm.set_nth_cov(array([[1.0, 0.0],[0.0, 1.0]]), 2)

est_em_gmm.set_coef(array([0.3333, 0.3333, 0.3334]))

print est_em_gmm.train_em(min_cov, max_em_iter, min_change)

#plot real GMM, data and both estimated GMMs
min_x_gen=min(min(generated[[0]]))-0.1
max_x_gen=max(max(generated[[0]]))+0.1
min_y_gen=min(min(generated[[1]]))-0.1
max_y_gen=max(max(generated[[1]]))+0.1

plot_real=empty(0)
plot_est_smem=empty(0)
plot_est_em=empty(0)

for i in arange(min_x_gen, max_x_gen, 0.05):
    for j in arange(min_y_gen, max_y_gen, 0.05):
        plot_real=append(plot_real, array([exp(real_gmm.cluster(array([i, j]))[3])]))
        plot_est_smem=append(plot_est_smem, array([exp(est_smem_gmm.cluster(array([i, j]))[3])]))
        plot_est_em=append(plot_est_em, array([exp(est_em_gmm.cluster(array([i, j]))[3])]))

plot_real=reshape(plot_real, (arange(min_x_gen, max_x_gen, 0.05).shape[0], arange(min_y_gen, max_y_gen, 0.05).shape[0]))
plot_est_smem=reshape(plot_est_smem, (arange(min_x_gen, max_x_gen, 0.05).shape[0], arange(min_y_gen, max_y_gen, 0.05).shape[0]))
plot_est_em=reshape(plot_est_em, (arange(min_x_gen, max_x_gen, 0.05).shape[0], arange(min_y_gen, max_y_gen, 0.05).shape[0]))

real_plot=contour(arange(min_x_gen, max_x_gen, 0.05), arange(min_y_gen, max_y_gen, 0.05), plot_real.transpose(), colors="b")
est_smem_plot=contour(arange(min_x_gen, max_x_gen, 0.05), arange(min_y_gen, max_y_gen, 0.05), plot_est_smem.transpose(), colors="r")
est_em_plot=contour(arange(min_x_gen, max_x_gen, 0.05), arange(min_y_gen, max_y_gen, 0.05), plot_est_em.transpose(), colors="g")
real_scatter=scatter(generated[[0]], generated[[1]], c="gray")
legend((real_plot.collections[0], est_em_plot.collections[0], est_smem_plot.collections[0]), ("Real GMM", "Estimated EM GMM", "Estimated SMEM GMM"))

connect('key_press_event', util.quit)
show()
