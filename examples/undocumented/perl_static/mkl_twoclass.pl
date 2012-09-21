from tools.load import LoadMatrix
from sg import sg
from numpy import *
num=100
labelstrain=concatenate((-ones([1,num]), ones([1,num])),1)[0]
featuretrain=concatenate((random.normal(size=(2,num))-1,random.normal(size=(2,num))+1),1)
parameter_list=[[1.,labelstrain,featuretrain],
				[1.,labelstrain,featuretrain]]

def mkl_twoclass (weight=1.,
		labels=labelstrain,features=featuretrain):

	sg('c', 10.)
	sg('new_classifier', 'MKL_CLASSIFICATION')

	sg('set_labels', 'TRAIN', labels)
	sg('add_features', 'TRAIN', features)
	sg('add_features', 'TRAIN', features)
	sg('add_features', 'TRAIN', features)

	sg('set_kernel', 'COMBINED', 100)
	sg('add_kernel', weight, 'GAUSSIAN', 'REAL', 100, 100.)
	sg('add_kernel', weight, 'GAUSSIAN', 'REAL', 100, 10.)
	sg('add_kernel', weight, 'GAUSSIAN', 'REAL', 100, 1.)
	sg('train_classifier')
	[bias, alphas]=sg('get_svm');

	km=sg('get_kernel_matrix', 'TEST')
	return km

if __name__=='__main__':
	print('MKL_TWOCLASS')
	mkl_twoclass(*parameter_list[0])

