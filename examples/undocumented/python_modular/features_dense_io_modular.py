#!/usr/bin/env python
parameter_list=[[]]

def labels_io_modular():
	from modshogun import RealFeatures, CSVFile, C_ORDER
	feats=RealFeatures()
	f=CSVFile("../data/fm_train_real.dat","r")
	#f.set_order(C_ORDER)
	f.set_delimiter(" ")
	feats.load(f)
	return feats

if __name__=='__main__':
	print('Dense Real Features IO')
	labels_io_modular(*parameter_list[0])
