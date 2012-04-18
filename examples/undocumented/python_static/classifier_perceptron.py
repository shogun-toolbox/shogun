from tools.load import LoadMatrix
from sg import sg
lm=LoadMatrix()

traindat=lm.load_numbers('../data/fm_train_real.dat')
testdat=lm.load_numbers('../data/fm_test_real.dat')
train_label=lm.load_labels('../data/label_train_twoclass.dat')
parameter_list=[[traindat,testdat, train_label],
		[traindat,testdat,train_label]]

def classifier_perceptron (fm_train_real=traindat,fm_test_real=testdat,
			label_train_twoclass=train_label):

	sg('set_features', 'TRAIN', fm_train_real)
	sg('set_labels', 'TRAIN', label_train_twoclass)
	sg('new_classifier', 'PERCEPTRON')
	# often does not converge, mind your data!
	sg('train_classifier')

	sg('set_features', 'TEST', fm_test_real)
	result=sg('classify')
	return result
if __name__=='__main__':
	print('Perceptron')
	classifier_perceptron(*parameter_list[0])
