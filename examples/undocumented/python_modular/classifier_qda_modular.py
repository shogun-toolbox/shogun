#!/usr/bin/env python
from tools.load import LoadMatrix
lm = LoadMatrix()

traindat = lm.load_numbers('../data/fm_train_real.dat')
testdat = lm.load_numbers('../data/fm_test_real.dat')
label_traindat = lm.load_labels('../data/label_train_multiclass.dat')

parameter_list = [[traindat, testdat, label_traindat, 1e-4, False], \
		  [traindat, testdat, label_traindat, 1e-4, True]]

def classifier_qda_modular (fm_train_real=traindat, fm_test_real=testdat, label_train_twoclass=label_traindat, tolerance=1e-4, store_covs=False):
	from modshogun import RealFeatures, MulticlassLabels
	from modshogun import QDA

	feats_train = RealFeatures(fm_train_real)
	feats_test  = RealFeatures(fm_test_real)

	labels = MulticlassLabels(label_train_twoclass)

	qda = QDA(feats_train, labels, tolerance, store_covs)
	qda.train()

	qda.apply(feats_test).get_labels()
	qda.set_features(feats_test)
	return qda, qda.apply().get_labels()

if __name__=='__main__':
	print('QDA')
	classifier_qda_modular(*parameter_list[0])
