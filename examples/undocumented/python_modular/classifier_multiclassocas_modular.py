from tools.load import LoadMatrix
lm=LoadMatrix()

traindat = lm.load_numbers('../data/fm_train_real.dat')
testdat = lm.load_numbers('../data/fm_test_real.dat')
label_traindat = lm.load_labels('../data/label_train_multiclass.dat')

parameter_list = [[traindat,testdat,label_traindat,2.1,1,1e-5],[traindat,testdat,label_traindat,2.2,1,1e-5]]

def classifier_multiclassocas_modular (fm_train_real=traindat,fm_test_real=testdat,label_train_multiclass=label_traindat,width=2.1,C=1,epsilon=1e-5):
	from shogun.Features import RealFeatures, Labels
	from shogun.Classifier import MulticlassOCAS

	feats_train=RealFeatures(fm_train_real)
	feats_test=RealFeatures(fm_test_real)

	labels=Labels(label_train_multiclass)

	classifier = MulticlassOCAS(C,feats_train,labels)
	classifier.train()

	out = classifier.apply(feats_test).get_labels()
	return out

if __name__=='__main__':
	print 'MulticlassOCAS'
	classifier_multiclassocas_modular(*parameter_list[0])
