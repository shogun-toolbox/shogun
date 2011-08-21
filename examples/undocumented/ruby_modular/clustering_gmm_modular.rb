# this was trancekoded by the awesome trancekoder
# ...and fixifikated by the awesum fixifikator
require 'modshogun'
require 'pp'

Math_init_random(5)

# *** real_gmm=GMM(2,0)
real_gmm=Modshogun::GMM.new
real_gmm.set_features(2,0)

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

def clustering_gmm_modular(fm_train=generated,n=2,min_cov=1e-9,max_iter=1000,min_change=1e-9,cov_type=0)


	Math_init_random(5)

# *** 	feat_train=RealFeatures(generated)
	feat_train=Modshogun::RealFeatures.new
	feat_train.set_features(generated)

# *** 	est_gmm=GMM(n, cov_type)
	est_gmm=Modshogun::GMM.new
	est_gmm.set_features(n, cov_type)
	est_gmm.train(feat_train)
	est_gmm.train_em(min_cov, max_iter, min_change)

	return est_gmm


end
if __FILE__ == $0
	puts 'GMM'
	clustering_gmm_modular(*parameter_list[0])


end
