# this was trancekoded by the awesome trancekoder
require 'narray'
require 'modshogun'
require 'load'
require 'pp'
traindat = LoadMatrix.load_numbers('../data/fm_train_real.dat')
testdat = LoadMatrix.load_numbers('../data/fm_test_real.dat')
label_traindat = LoadMatrix.load_labels('../data/label_train_multiclass.dat')

parameter_list = [[traindat,testdat,label_traindat,3],[traindat,testdat,label_traindat,3]]

def classifier_knn_modular(fm_train_real=traindat,fm_test_real=testdat,label_train_multiclass=label_traindat, k=3 )

	feats_train=RealFeatures(fm_train_real)
	feats_test=RealFeatures(fm_test_real)
	distance=EuclidianDistance(feats_train, feats_train)


	labels=Labels(label_train_multiclass)

	knn=KNN(k, distance, labels)
	knn_train = knn.train()
	output=knn.apply(feats_test).get_labels()
	multiple_k=knn.classify_for_multiple_k()
	return knn,knn_train,output,multiple_k


end
if __FILE__ == $0
	print 'KNN'
	classifier_knn_modular(*parameter_list[0])

end
