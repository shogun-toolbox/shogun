# this was trancekoded by the awesome trancekoder
require 'narray'
require 'modshogun'
require 'load'
require 'pp'

traindat = LoadMatrix.load_numbers('../data/fm_train_real.dat')
testdat = LoadMatrix.load_numbers('../data/fm_test_real.dat')
label_traindat = LoadMatrix.load_labels('../data/label_train_twoclass.dat')


parameter_list = [[traindat,testdat,label_traindat,2.1,1,1e-5,1e-2], \
                 [traindat,testdat,label_traindat,2.1,1,1e-5,1e-2]]


def regression_libsvr_modular(fm_train=traindat,fm_test=testdat,label_train=label_traindat,\
				       width=2.1,C=1,epsilon=1e-5,tube_epsilon=1e-2):


	feats_train=RealFeatures(fm_train)
	feats_test=RealFeatures(fm_test)

	kernel=GaussianKernel(feats_train, feats_train, width)
	labels=Labels(label_train)

	svr=LibSVR(C, epsilon, kernel, labels)
	svr.set_tube_epsilon(tube_epsilon)
	svr.train()

	kernel.init(feats_train, feats_test)
	out1=svr.apply().get_labels()
	out2=svr.apply(feats_test).get_labels()

	return out1,out2,kernel


end
if __FILE__ == $0
	print 'LibSVR'
	regression_libsvr_modular(*parameter_list[0])

end
