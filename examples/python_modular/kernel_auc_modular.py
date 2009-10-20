###########################################################################
# kernel can be used to maximize AUC instead of margin in SVMs 
###########################################################################

def auc ():
	print 'AUC'

	from shogun.Kernel import GaussianKernel, AUCKernel
	from shogun.Features import RealFeatures, Labels

	feats_train=RealFeatures(fm_train_real)
	width=1.7
	subkernel=GaussianKernel(feats_train, feats_train, width)

	kernel=AUCKernel(0, subkernel)
	kernel.setup_auc_maximization( Labels(label_train_real) )
	km_train=kernel.get_kernel_matrix()

if __name__=='__main__':
	from tools.load import LoadMatrix
	from numpy import double
	lm=LoadMatrix()
	fm_train_real=double(lm.load_numbers('../data/fm_train_real.dat'))
	label_train_real=lm.load_labels('../data/label_train_twoclass.dat')
	auc()
