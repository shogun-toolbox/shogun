#!/usr/bin/env python
from tools.load import LoadMatrix
lm=LoadMatrix()

traindat = lm.load_dna('../data/fm_train_dna.dat')
testdat = lm.load_dna('../data/fm_test_dna.dat')
label_traindat = lm.load_labels('../data/label_train_dna.dat')

parameter_list = [[traindat,testdat,label_traindat,1.1,1e-5,1],[traindat,testdat,label_traindat,1.2,1e-5,1]]

def classifier_svmlight_modular (fm_train_dna=traindat,fm_test_dna=testdat,label_train_dna=label_traindat,C=1.2,epsilon=1e-5,num_threads=1):
	from modshogun import StringCharFeatures, BinaryLabels, DNA
	from modshogun import WeightedDegreeStringKernel
	try:
		from modshogun import SVMLight
	except ImportError:
		print('No support for SVMLight available.')
		return

	feats_train=StringCharFeatures(DNA)
	feats_train.set_features(fm_train_dna)
	feats_test=StringCharFeatures(DNA)
	feats_test.set_features(fm_test_dna)
	degree=20

	kernel=WeightedDegreeStringKernel(feats_train, feats_train, degree)

	labels=BinaryLabels(label_train_dna)

	svm=SVMLight(C, kernel, labels)
	svm.set_epsilon(epsilon)
	svm.parallel.set_num_threads(num_threads)
	svm.train()

	kernel.init(feats_train, feats_test)
	svm.apply().get_labels()
	return kernel
if __name__=='__main__':
	print('SVMLight')
	classifier_svmlight_modular(*parameter_list[0])
