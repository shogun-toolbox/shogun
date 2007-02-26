from numpy.random import rand
from numpy import sign
from shogun.Features import RealFeatures,Labels
from shogun.Kernel import GaussianKernel,M_DEBUG
from shogun.Classifier import LibSVM


def fun():
	C=1
	lab=Labels(sign(rand(10)-0.5))
	lab.io.set_loglevel(M_DEBUG)
	feat = RealFeatures(rand(5,10))
	feat.io.set_loglevel(M_DEBUG)
	gk=GaussianKernel(feat,feat, 1.0, 10)
	gk.io.set_loglevel(M_DEBUG)
	print gk.ref_count()

	svm=LibSVM(C, gk, lab)
	svm.io.set_loglevel(M_DEBUG)

	print "lab:" + `lab.ref_count()`
	print "feat:" + `feat.ref_count()`
	print "gk:" + `gk.ref_count()`
	#del svm
	print "lab:" + `lab.ref_count()`
	print "feat:" + `feat.ref_count()`
	print "gk:" + `gk.ref_count()`
	k2=svm.get_kernel()
	print "gk:" + `gk.ref_count()`
	print "k2:" + `k2.ref_count()`
	s=gk
	del gk
	print "s:" + `s.ref_count()`
	svm.train()
	del svm
	#del s
	print s.ref_count()
	print "k2:" + `k2.ref_count()`

	test_feat = RealFeatures(rand(5,100))
	s.init(feat, test_feat)
	print "inside"

print "pre"
fun()
print "post"
