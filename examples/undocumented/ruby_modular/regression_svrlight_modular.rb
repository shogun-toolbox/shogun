# this was trancekoded by the awesome trancekoder
require 'narray'
require 'modshogun'
require 'load'
require 'pp'
###########################################################################
# svm light based support vector regression
###########################################################################

traindat = LoadMatrix.load_numbers('../data/fm_train_real.dat')
testdat = LoadMatrix.load_numbers('../data/fm_test_real.dat')
label_traindat = LoadMatrix.load_labels('../data/label_train_twoclass.dat')

parameter_list = [[traindat,testdat,label_traindat,1.2,1,1e-5,1e-2,1],[traindat,testdat,label_traindat,2.3,0.5,1e-5,1e-6,1]]

def regression_svrlight_modular(fm_train=traindat,fm_test=testdat,label_train=label_traindat, \
				    width=1.2,C=1,epsilon=1e-5,tube_epsilon=1e-2,num_threads=3):


	try:
	except ImportError:
		print 'No support for SVRLight available.'
		return

	feats_train=RealFeatures(fm_train)
	feats_test=RealFeatures(fm_test)

	kernel=GaussianKernel(feats_train, feats_train, width)

	labels=Labels(label_train)

	svr=SVRLight(C, epsilon, kernel, labels)
	svr.set_tube_epsilon(tube_epsilon)
	svr.parallel.set_num_threads(num_threads)
	svr.train()

	kernel.init(feats_train, feats_test)
	out = svr.apply().get_labels()
	
	return out, kernel 


end
if __FILE__ == $0
	print 'SVRLight'
	regression_svrlight_modular(*parameter_list[0])

end
