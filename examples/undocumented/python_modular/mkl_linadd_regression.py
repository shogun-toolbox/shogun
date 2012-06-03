from shogun.Features import CombinedFeatures, RealFeatures, BinaryLabels, RegressionLabels
from shogun.Kernel import CombinedKernel, PolyKernel, CustomKernel
from shogun.Classifier import MKLClassification, MKLRegression
from modshogun import *
from tools.load import LoadMatrix
lm=LoadMatrix()

traindat = lm.load_numbers('../data/fm_train_real.dat')
testdat = lm.load_numbers('../data/fm_test_real.dat')
label_traindat = lm.load_labels('../data/label_train_twoclass.dat')
strings1 = ["hey"]*92
strings2 = ["yeh"]*92

parameter_list = [[traindat,testdat,label_traindat],[traindat,testdat,label_traindat]]
#    fm_train_real.shape
#    fm_test_real.shape
#    combined_custom()

def mkl_binclass_modular (fm_train_real=traindat,fm_test_real=testdat,fm_label_twoclass = label_traindat):

	sc1 = StringCharFeatures(strings1, RAWBYTE)
	sc2 = StringCharFeatures(strings2, RAWBYTE)

	sfeats1 = StringWordFeatures(RAWBYTE)
	sfeats1.obtain_from_char(sc1,0,2,0,False)
	skernel1 = CommWordStringKernel(10,False)
	skernel1.init(sfeats1, sfeats1)
	sfeats2 = StringWordFeatures(RAWBYTE)
	sfeats2.obtain_from_char(sc2,0,2,0,False)
	skernel2 = CommWordStringKernel(10,False)
	skernel2.init(sfeats2, sfeats2)

	ffeats = RealFeatures(traindat)
	fkernel = LinearKernel(ffeats,ffeats)

	fffeats = RealFeatures(traindat)
	ffkernel = GaussianKernel(fffeats,fffeats,1.0)

# COMBINING LINADD FEATURES/KERNELS LEAD TO FAIL
	feats_train = CombinedFeatures()
	feats_train.append_feature_obj(ffeats)
	#feats_train.append_feature_obj(fffeats)
	feats_train.append_feature_obj(sfeats2)

	print feats_train.get_num_vectors()

	kernel = CombinedKernel()
	kernel.append_kernel(fkernel)
	#kernel.append_kernel(ffkernel)
	kernel.append_kernel(skernel2)
	kernel.init(feats_train, feats_train)

	labels = RegressionLabels(fm_label_twoclass)
	mkl = MKLRegression()

	mkl.set_mkl_norm(1) #2,3
	mkl.set_C(1, 1)
	mkl.set_kernel(kernel)
	mkl.set_labels(labels)

	mkl.io.enable_file_and_line()
	mkl.io.set_loglevel(MSG_DEBUG)
	#mkl.train()

if __name__=='__main__':
	mkl_binclass_modular (*parameter_list[0])
