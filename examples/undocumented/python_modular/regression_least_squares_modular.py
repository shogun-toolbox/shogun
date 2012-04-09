###########################################################################
# kernel ridge regression
###########################################################################
from numpy import array, load
from numpy.random import seed, rand
from tools.load import LoadMatrix
lm=LoadMatrix()

# traindat = lm.load_numbers('../data/fm_train_real.dat')
# testdat = lm.load_numbers('../data/fm_test_real.dat')
# label_traindat = lm.load_labels('../data/label_train_twoclass.dat')
traindat = load('/home/pluskid/work/gsoc/sandbox/cover-tree/train_matrix_prewitt.npy')
testdat = load('/home/pluskid/work/gsoc/sandbox/cover-tree/test_matrix_prewitt.npy')
label_traindat = load('/home/pluskid/work/gsoc/sandbox/cover-tree/train_labels_prewitt.npy')

parameter_list = [[traindat[:,1:1000],testdat[:,1:1000],label_traindat[1:1000]]]

def regression_least_squares_modular (fm_train=traindat,fm_test=testdat,label_train=label_traindat,tau=1e-6):

	from shogun.Features import Labels, RealFeatures
	from shogun.Kernel import GaussianKernel
	from shogun.Regression import LeastSquaresRegression

	ls=LeastSquaresRegression(RealFeatures(traindat), Labels(label_train))
	ls.train()
	out = ls.apply(RealFeatures(fm_test)).get_labels()
	return out,ls

if __name__=='__main__':
	print('LeastSquaresRegression')
	regression_least_squares_modular(*parameter_list[0])
