from tools.load import LoadMatrix
lm=LoadMatrix()

traindat = lm.load_numbers('../data/fm_train_real.dat')
testdat = lm.load_numbers('../data/fm_test_real.dat')
label_traindat = lm.load_labels('../data/label_train_twoclass.dat')

parameter_list = [[traindat,testdat,label_traindat,3,1],[traindat,testdat,label_traindat,4,1]]

def classifier_lda_modular (fm_train_real=traindat,fm_test_real=testdat,label_train_twoclass=label_traindat,gamma=3,num_threads=1):
	from shogun.Features import RealFeatures, Labels
	from shogun.Classifier import LDA

	feats_train=RealFeatures(fm_train_real)
	feats_test=RealFeatures(fm_test_real)

	labels=Labels(label_train_twoclass)

	lda=LDA(gamma, feats_train, labels)
	lda.train()

	lda.get_bias()
	lda.get_w()
	lda.set_features(feats_test)
	lda.apply().get_labels()
	return lda,lda.apply().get_labels()

if __name__=='__main__':
	print('LDA')
	classifier_lda_modular(*parameter_list[0])
