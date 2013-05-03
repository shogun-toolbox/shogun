#!/usr/bin/env python
from numpy import array,hstack
from numpy.random import seed, rand
from tools.load import LoadMatrix
lm=LoadMatrix()

traindat = lm.load_numbers('../data/fm_train_real.dat')
testdat = lm.load_numbers('../data/fm_test_real.dat')
label_traindat = lm.load_labels('../data/label_train_twoclass.dat')

parameter_list = [[traindat,testdat,label_traindat]]

def transfer_multitask_trace_logistic_regression (fm_train=traindat,fm_test=testdat,label_train=label_traindat):

	from modshogun import BinaryLabels, RealFeatures, Task, TaskGroup, MultitaskTraceLogisticRegression

	features = RealFeatures(hstack((traindat,traindat)))
	labels = BinaryLabels(hstack((label_train,label_train)))

	n_vectors = features.get_num_vectors()
	task_one = Task(0,n_vectors//2)
	task_two = Task(n_vectors//2,n_vectors)
	task_group = TaskGroup()
	task_group.append_task(task_one)
	task_group.append_task(task_two)

	mtlr = MultitaskTraceLogisticRegression(0.1,features,labels,task_group)
	mtlr.set_tolerance(1e-2) # use 1e-2 tolerance
	mtlr.set_max_iter(10)
	mtlr.train()
	mtlr.set_current_task(0)
	out = mtlr.apply_regression().get_labels()

	return out

if __name__=='__main__':
	print('TransferMultitaskTraceLogisticRegression')
	transfer_multitask_trace_logistic_regression(*parameter_list[0])
