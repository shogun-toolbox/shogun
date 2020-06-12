#!/usr/bin/env python
import shogun as sg
parameter_list=[[]]

def labels_io():
	f=sg.read_csv("../data/label_train_regression.dat","r")
	lab = sg.create_features(f)
	#print lab.get_labels()
	return lab

if __name__=='__main__':
	print('Labels IO')
	labels_io(*parameter_list[0])
