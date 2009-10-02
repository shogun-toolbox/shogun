###########################################################################
# kernel ridge regression
###########################################################################

def krr ():
	print 'KRR'
	from shogun.Features import Labels, RealFeatures
	from shogun.Kernel import GaussianKernel
	from shogun.Regression import KRR

	feats_train=RealFeatures(fm_train)
	feats_test=RealFeatures(fm_test)
	width=0.8
	kernel=GaussianKernel(feats_train, feats_train, width)

	C=0.9
	tau=1e-6
	labels=Labels(label_train)

	krr=KRR(tau, kernel, labels)
	krr.train()

	kernel.init(feats_train, feats_test)
	krr.classify().get_labels()

if __name__=='__main__':
	from numpy import array
	from numpy.random import seed, rand
	from tools.load import LoadMatrix
	lm=LoadMatrix()
	fm_train=lm.load_numbers('../data/fm_train_real.dat')
	fm_test=lm.load_numbers('../data/fm_test_real.dat')
	label_train=lm.load_labels('../data/label_train_twoclass.dat')
	krr()
