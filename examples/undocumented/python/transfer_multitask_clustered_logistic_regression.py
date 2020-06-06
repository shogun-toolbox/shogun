#!/usr/bin/env python
from numpy import array,hstack,sin,cos
from numpy.random import seed, rand
from tools.load import LoadMatrix
lm=LoadMatrix()

traindat = lm.load_numbers('../data/fm_train_real.dat')
testdat = lm.load_numbers('../data/fm_test_real.dat')
label_traindat = lm.load_labels('../data/label_train_twoclass.dat')

parameter_list = [[traindat,testdat,label_traindat]]

def transfer_multitask_clustered_logistic_regression (fm_train=traindat,fm_test=testdat,label_train=label_traindat):
	import shogun as sg
	try:
		sg.create_machine("MultitaskClusteredLogisticRegression")
	except ImportError:
		print("MultitaskClusteredLogisticRegression not available")
		exit()

	features = sg.create_features(hstack((traindat,sin(traindat),cos(traindat))))
	labels = sg.BinaryLabels(hstack((label_train,label_train,label_train)))

	n_vectors = features.get_num_vectors()
	task_one = sg.Task(0,n_vectors//3)
	task_two = sg.Task(n_vectors//3,2*n_vectors//3)
	task_three = sg.Task(2*n_vectors//3,n_vectors)
	task_group = sg.TaskGroup()
	task_group.append_task(task_one)
	task_group.append_task(task_two)
	task_group.append_task(task_three)

	mtlr = sg.MultitaskClusteredLogisticRegression(1.0,100.0,features,labels,task_group,2)
	#mtlr.io.set_loglevel(MSG_DEBUG)
	mtlr.set_tolerance(1e-3) # use 1e-2 tolerance
	mtlr.set_max_iter(100)
	mtlr.train()
	mtlr.set_current_task(0)
	#print mtlr.get_w()
	out = mtlr.apply_regression().get("labels")

	return out

if __name__=='__main__':
	print('TransferMultitaskClusteredLogisticRegression')
	transfer_multitask_clustered_logistic_regression(*parameter_list[0])
