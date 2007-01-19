from numpy.random import rand
from shogun.Features import RealFeatures
from shogun.Kernel import GaussianKernel

feat = RealFeatures(rand(5,10))
gk=GaussianKernel(feat,feat, 1.0, 10)
km=gk.get_kernel_matrix()


test_feat = RealFeatures(rand(5,100))
gk.init(feat, test_feat, True)
test_km=gk.get_kernel_matrix()
