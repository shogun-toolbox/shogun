###########################################################################
# svm light based support vector regression
###########################################################################

def svr_light ():
	print 'SVRLight'
	from shogun.Features import Labels, RealFeatures
	from shogun.Kernel import GaussianKernel
	try:
		from shogun.Regression import SVRLight
	except ImportError:
		print 'No support for SVRLight available.'
		return

	feats_train=RealFeatures(fm_train)
	feats_test=RealFeatures(fm_test)
	width=2.1
	kernel=GaussianKernel(feats_train, feats_train, width)

	C=1
	epsilon=1e-5
	tube_epsilon=1e-2
	num_threads=3
	labels=Labels(label_train)

	svr=SVRLight(C, epsilon, kernel, labels)
	svr.set_tube_epsilon(tube_epsilon)
	svr.parallel.set_num_threads(num_threads)
	svr.train()

	kernel.init(feats_train, feats_test)
	svr.classify().get_labels()

if __name__=='__main__':
	from numpy import array
	from numpy.random import seed, rand
	from tools.load import LoadMatrix
	lm=LoadMatrix()
	fm_train=lm.load_numbers('../data/fm_train_real.dat')
	fm_test=lm.load_numbers('../data/fm_test_real.dat')
	label_train=lm.load_labels('../data/label_train_twoclass.dat')
	svr_light()
