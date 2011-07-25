from numpy import array, append
from shogun.Distribution import GMM
from shogun.Library import Math_init_random

Math_init_random(5)

real_gmm=GMM(2,0)

real_gmm.set_nth_mean(array([1.0, 1.0]), 0)
real_gmm.set_nth_mean(array([-1.0, -1.0]), 1)

real_gmm.set_nth_cov(array([[1.0, 0.2],[0.2, 0.1]]), 0)
real_gmm.set_nth_cov(array([[0.3, 0.1],[0.1, 1.0]]), 1)

real_gmm.set_coef(array([0.3, 0.7]))

generated=array([real_gmm.sample()])
for i in range(199):
    generated=append(generated, array([real_gmm.sample()]), axis=0)

generated=generated.transpose()

parameter_list = [[generated,2,1e-9,1000,1e-9,0]]

def clustering_gmm_modular (fm_train=generated,n=2,min_cov=1e-9,max_iter=1000,min_change=1e-9,cov_type=0):

	from shogun.Distribution import GMM
	from shogun.Features import RealFeatures
	from shogun.Library import Math_init_random

	Math_init_random(5)

	feat_train=RealFeatures(generated)

	est_gmm=GMM(n, cov_type)
	est_gmm.train(feat_train)
	est_gmm.train_em(min_cov, max_iter, min_change)

	return est_gmm

if __name__=='__main__':
	print 'GMM'
	clustering_gmm_modular(*parameter_list[0])

