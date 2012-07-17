from numpy import array
from numpy.random import seed, rand
from tools.load import LoadMatrix
lm=LoadMatrix()

traindat = lm.load_numbers('../data/fm_train_real.dat')
testdat = lm.load_numbers('../data/fm_test_real.dat')
label_traindat = lm.load_labels('../data/label_train_twoclass.dat')

parameter_list = [[traindat,testdat,label_traindat]]

def transfer_multitask_leastsquares_regression(fm_train=traindat,fm_test=testdat,label_train=label_traindat):

	from modshogun import RegressionLabels, RealFeatures, IndexBlock, IndexBlockGroup, MultitaskLSRegression

	features = RealFeatures(traindat)
	labels = RegressionLabels(label_train)

	n_vectors = features.get_num_vectors()
	task_one = IndexBlock(0,n_vectors/2)
	task_two = IndexBlock(n_vectors/2,n_vectors)
	task_group = IndexBlockGroup()
	task_group.add_block(task_one)
	task_group.add_block(task_two)

	mtlsr = MultitaskLSRegression(0.1,features,labels,task_group)
	mtlsr.set_regularization(1) # use regularization ratio
	mtlsr.set_tolerance(1e-2) # use 1e-2 tolerance
	mtlsr.train()
	mtlsr.set_current_task(0)
	out = mtlsr.apply_regression().get_labels()
	return out

if __name__=='__main__':
	print('TransferMultitaskLeastSquaresRegression')
	transfer_multitask_leastsquares_regression(*parameter_list[0])
