#!/usr/bin/env python
traindat = '../data/fm_train_real.dat'
testdat = '../data/fm_test_real.dat'
label_traindat = '../data/label_train_multiclass.dat'

parameter_list = [[traindat, testdat, label_traindat, 1e-4, False], \
		  [traindat, testdat, label_traindat, 1e-4, True]]

def classifier_qda_modular (train_fname=traindat, test_fname=testdat, label_fname=label_traindat, tolerance=1e-4, store_covs=False):
	from modshogun import RealFeatures, MulticlassLabels
	from modshogun import QDA, CSVFile

	feats_train = RealFeatures(fm_train_real)
	feats_test  = RealFeatures(fm_test_real)

	feats_train=RealFeatures(CSVFile(train_fname))
	feats_test=RealFeatures(CSVFile(test_fname))
	labels=BinaryLabels(CSVFile(label_fname))

	qda = QDA(feats_train, labels, tolerance, store_covs)
	qda.train()

	qda.apply(feats_test).get_labels()
	qda.set_features(feats_test)
	return qda, qda.apply().get_labels()

if __name__=='__main__':
	print('QDA')
	classifier_qda_modular(*parameter_list[0])
