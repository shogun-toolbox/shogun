###########################################################################
# kernel ridge regression
###########################################################################
from numpy import array
from numpy.random import seed, rand
from tools.load import LoadMatrix
lm=LoadMatrix()

traindat = lm.load_numbers('../data/fm_train_real.dat')
testdat = lm.load_numbers('../data/fm_test_real.dat')
label_traindat = lm.load_labels('../data/label_train_twoclass.dat')


parameter_list = [[traindat,testdat,label_traindat,0.8,1e-6],[traindat,testdat,label_traindat,0.9,1e-7]]

def regression_kernel_ridge_modular (fm_train=traindat,fm_test=testdat,label_train=label_traindat,width=0.8,tau=1e-6):

	from shogun.Features import Labels, RealFeatures
	from shogun.Kernel import GaussianKernel
	from shogun.Regression import KernelRidgeRegression

	feats_train=RealFeatures(fm_train)
	feats_test=RealFeatures(fm_test)

	kernel=GaussianKernel(feats_train, feats_train, width)

	labels=Labels(label_train)

	krr=KernelRidgeRegression(tau, kernel, labels)
	krr.train(feats_train)

	kernel.init(feats_train, feats_test)
	out = krr.apply().get_labels()
	return out,kernel,krr

# equivialent shorter version
def krr_short ():
	print('KRR_short')
	from shogun.Features import Labels, RealFeatures
	from shogun.Kernel import GaussianKernel
	from shogun.Regression import KernelRidgeRegression

	width=0.8; tau=1e-6
	krr=KernelRidgeRegression(tau, GaussianKernel(0, width), Labels(label_train))
	krr.train(RealFeatures(fm_train))
	out = krr.apply(RealFeatures(fm_test)).get_labels()

	return krr,out

if __name__=='__main__':
	print('KRR')
	regression_kernel_ridge_modular(*parameter_list[0])
