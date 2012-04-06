###########################################################################
# kernel can be used to maximize AUC instead of margin in SVMs 
###########################################################################
from tools.load import LoadMatrix
from numpy import double
lm=LoadMatrix()

traindat = double(lm.load_numbers('../data/fm_train_real.dat'))
testdat = lm.load_labels('../data/label_train_twoclass.dat')
parameter_list = [[traindat,testdat,1.7], [traindat,testdat,1.6]]


def kernel_auc_modular(fm_train_real=traindat,label_train_real=testdat,width=1.7):


	from shogun.Kernel import GaussianKernel, AUCKernel
	from shogun.Features import RealFeatures, Labels

	feats_train=RealFeatures(fm_train_real)

	subkernel=GaussianKernel(feats_train, feats_train, width)

	kernel=AUCKernel(0, subkernel)
	kernel.setup_auc_maximization( Labels(label_train_real) )
	km_train=kernel.get_kernel_matrix()
	return kernel

if __name__=='__main__':
	print('AUC')
	kernel_auc_modular(*parameter_list[0])
