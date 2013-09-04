#!/usr/bin/env python
traindat = '../data/fm_train_real.dat'
testdat = '../data/fm_test_real.dat'
label_traindat = '../data/label_train_multiclass.dat'

parameter_list = [[traindat,testdat,label_traindat,0.9,1e-3],[traindat,testdat,label_traindat,0.8,1e-2]]

def classifier_liblinear_modular (train_fname, test_fname,
		label_traindat, C, epsilon):

	from modshogun import RealFeatures, SparseRealFeatures, BinaryLabels
	from modshogun import LibLinear, L2R_L2LOSS_SVC_DUAL
	from modshogun import Math_init_random
	Math_init_random(17)

	feats_train=RealFeatures(CSVFile(train_fname))
	feats_test=RealFeatures(CSVFile(test_fname))
	labels=BinaryLabels(CSVFile(label_fname))

	svm=LibLinear(C, feats_train, labels)
	svm.set_liblinear_solver_type(L2R_L2LOSS_SVC_DUAL)
	svm.set_epsilon(epsilon)
	svm.set_bias_enabled(True)
	svm.train()

	predictions = svm.apply(feats_test)
	return predictions, svm, predictions.get_labels()

if __name__=='__main__':
	print('LibLinear')
	classifier_liblinear_modular(*parameter_list[0])
