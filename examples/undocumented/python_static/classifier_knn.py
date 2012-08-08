from tools.load import LoadMatrix
from sg import sg
lm=LoadMatrix()


traindat=lm.load_numbers('../data/fm_train_real.dat')
testdat=lm.load_numbers('../data/fm_test_real.dat')
train_label=lm.load_labels('../data/label_train_multiclass.dat')
parameter_list=[[traindat,testdat, train_label,3],
		[traindat,testdat,train_label,4]]

def classifier_knn (fm_train_real=traindat,fm_test_real=testdat,
			label_train_multiclass=train_label,k=3):

	sg('set_features', 'TRAIN', fm_train_real)
	sg('set_labels', 'TRAIN', label_train_multiclass)
	sg('set_distance', 'EUCLIDEAN', 'REAL')
	sg('new_classifier', 'KNN')
	sg('train_classifier', k)

	sg('set_features', 'TEST', fm_test_real)
	result=sg('classify')
	return result

if __name__=='__main__':
	print('KNN')
	classifier_knn(*parameter_list[0])
