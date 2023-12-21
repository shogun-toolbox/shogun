#!/usr/bin/env python
from numpy import array,hstack
from numpy.random import seed, rand
from tools.load import LoadMatrix
lm=LoadMatrix()

traindat = lm.load_numbers('../data/fm_train_real.dat')
testdat = lm.load_numbers('../data/fm_test_real.dat')
label_traindat = lm.load_labels('../data/label_train_twoclass.dat')

parameter_list = [[traindat,testdat,label_traindat]]

def transfer_multitask_logistic_regression (fm_train=traindat,fm_test=testdat,label_train=label_traindat):
	from shogun import BinaryLabels, Task, TaskGroup
	try:
		from shogun import MultitaskLogisticRegression
	except ImportError:
		print("MultitaskLogisticRegression not available")
		exit()
	import shogun as sg

	features = sg.create_features(hstack((traindat,traindat)))
	labels = BinaryLabels(hstack((label_train,label_train)))

	n_vectors = features.get_num_vectors()
	task_one = Task(0,n_vectors//2)
	task_two = Task(n_vectors//2,n_vectors)
	task_group = TaskGroup()
	task_group.append_task(task_one)
	task_group.append_task(task_two)

	mtlr = MultitaskLogisticRegression(0.1,task_group)
	mtlr.set_regularization(1) # use regularization ratio
	mtlr.set_tolerance(1e-2) # use 1e-2 tolerance
	mtlr.train(features,labels)
	mtlr.set_current_task(0)
	out = mtlr.apply(features).get("labels")

	return out

if __name__=='__main__':
	print('TransferMultitaskLogisticRegression')
	transfer_multitask_logistic_regression(*parameter_list[0])
