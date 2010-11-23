###########################################################################
# kernel can be used to maximize AUC instead of margin in SVMs 
###########################################################################
from tools.load import LoadMatrix
from numpy import double
lm=LoadMatrix()

parameter_list = [[double(lm.load_numbers('../data/fm_train_real.dat')),lm.load_labels('../data/label_train_twoclass.dat'),1.7], [double(lm.load_numbers('../data/fm_train_real.dat')),lm.load_labels('../data/label_train_twoclass.dat'),1.8]]


def kernel_auc_modular(fm_train_real=double(lm.load_numbers('../data/fm_train_real.dat')),label_train_real=lm.load_labels('../data/label_train_twoclass.dat'),width=1.7):
	print 'AUC'

	from shogun.Kernel import GaussianKernel, AUCKernel
	from shogun.Features import RealFeatures, Labels
	width            = width
	fm_train_real    = fm_train_real
	label_train_real = label_train_real

	feats_train=RealFeatures(fm_train_real)

	subkernel=GaussianKernel(feats_train, feats_train, width)

	kernel=AUCKernel(0, subkernel)
	kernel.setup_auc_maximization( Labels(label_train_real) )
	km_train=kernel.get_kernel_matrix()
	print km_train

if __name__=='__main__':
	from tools.load import LoadMatrix
	from numpy import double
	lm=LoadMatrix()
	fm_train_real=double(lm.load_numbers('../data/fm_train_real.dat'))
	label_train_real=lm.load_labels('../data/label_train_twoclass.dat')
	kernel_auc_modular(*parameter_list[0])
