from tools.load import LoadMatrix
from sg import sg
lm=LoadMatrix()

traindat=lm.load_numbers('../data/fm_train_real.dat')
testdat=lm.load_numbers('../data/fm_test_real.dat')
trainlabel=lm.load_labels('../data/label_train_regression.dat')
parameter_list=[[traindat,testdat,trainlabel,10,2.1,1.2,1e-6],
		[traindat,testdat,trainlabel,11,2.3,1.3,1e-6]]

def regression_krr (fm_train=traindat,fm_test=testdat,
		label_train=trainlabel,size_cache=10,width=2.1,
		C=1.2,tau=1e-6):

	sg('set_features', 'TRAIN', fm_train)
	sg('set_kernel', 'GAUSSIAN', 'REAL', size_cache, width)

	sg('set_labels', 'TRAIN', label_train)

	sg('new_regression', 'KERNELRIDGEREGRESSION')
	sg('krr_tau', tau)
	sg('c', C)
	sg('train_regression')

	sg('set_features', 'TEST', fm_test)
	result=sg('classify')
	return result

if __name__=='__main__':
	print('KRR')
	regression_krr(*parameter_list[0])
