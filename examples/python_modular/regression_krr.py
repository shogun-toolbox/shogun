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

	tau=1e-6
	labels=Labels(label_train)

	krr=KRR(tau, kernel, labels)
	krr.train(feats_train)

	kernel.init(feats_train, feats_test)
	out = krr.classify().get_labels()
	return out

# equivialent shorter version
def krr_short ():
	print 'KRR_short'
	from shogun.Features import Labels, RealFeatures
	from shogun.Kernel import GaussianKernel
	from shogun.Regression import KRR

	width=0.8; tau=1e-6
	krr=KRR(tau, GaussianKernel(0, width), Labels(label_train))
	krr.train(RealFeatures(fm_train))
	out = krr.classify(RealFeatures(fm_test)).get_labels()
	return out

if __name__=='__main__':
	from numpy import array
	from numpy.random import seed, rand
	from tools.load import LoadMatrix
	lm=LoadMatrix()
	fm_train=lm.load_numbers('../data/fm_train_real.dat')
	fm_test=lm.load_numbers('../data/fm_test_real.dat')
	label_train=lm.load_labels('../data/label_train_twoclass.dat')
	out1=krr()
	out2=krr_short()
