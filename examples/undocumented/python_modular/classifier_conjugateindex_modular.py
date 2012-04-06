from tools.load import LoadMatrix
lm=LoadMatrix()

traindat = lm.load_numbers('../data/fm_train_real.dat')
testdat = lm.load_numbers('../data/fm_test_real.dat')
label_traindat = lm.load_labels('../data/label_train_multiclass.dat')

parameter_list = [[traindat,testdat,label_traindat],[traindat,testdat,label_traindat]]

def classifier_conjugateindex_modular (fm_train_real=traindat,fm_test_real=testdat,label_train_multiclass=label_traindat):
	from shogun.Features import RealFeatures, Labels
	from shogun.Classifier import ConjugateIndex

	feats_train = RealFeatures(fm_train_real)
	feats_test = RealFeatures(fm_test_real)

	labels = Labels(label_train_multiclass)

	ci = ConjugateIndex(feats_train, labels)
	ci.train()

	res = ci.apply(feats_test).get_labels()
	return ci, res

if __name__=='__main__':
	print('ConjugateIndex')
	classifier_conjugateindex_modular(*parameter_list[0])
