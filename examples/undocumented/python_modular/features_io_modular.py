#!/usr/bin/env python
from tools.load import LoadMatrix
lm=LoadMatrix()
data=lm.load_numbers('../data/fm_train_real.dat')
label=lm.load_numbers('../data/label_train_twoclass.dat')

parameter_list=[[data,label]]

def features_io_modular (fm_train_real, label_train_twoclass):
	import numpy
	from modshogun import SparseRealFeatures, RealFeatures, MulticlassLabels
	from modshogun import GaussianKernel
	from modshogun import CSVFile, BinaryFile, HDF5File

	feats=SparseRealFeatures(fm_train_real)
	feats2=SparseRealFeatures()

	f=BinaryFile("fm_train_sparsereal.bin","w")
	feats.save(f)

	f=CSVFile("fm_train_sparsereal.ascii","w")
	feats.save(f)

	f=BinaryFile("fm_train_sparsereal.bin")
	feats2.load(f)

	f=CSVFile("fm_train_sparsereal.ascii")
	feats2.load(f)

	feats=RealFeatures(fm_train_real)
	feats2=RealFeatures()

	f=BinaryFile("fm_train_real.bin","w")
	feats.save(f)

	f=HDF5File("fm_train_real.h5","w", "/data/doubles")
	feats.save(f)

	f=CSVFile("fm_train_real.ascii","w")
	feats.save(f)

	f=BinaryFile("fm_train_real.bin")
	feats2.load(f)
	#print("diff binary", numpy.max(numpy.abs(feats2.get_feature_matrix().flatten()-fm_train_real.flatten())))

	f=CSVFile("fm_train_real.ascii")
	feats2.load(f)
	#print("diff ascii", numpy.max(numpy.abs(feats2.get_feature_matrix().flatten()-fm_train_real.flatten())))

	lab=MulticlassLabels(numpy.array([0.0,1.0,2.0,3.0]))
	lab2=MulticlassLabels()
	f=CSVFile("label_train_twoclass.ascii","w")
	lab.save(f)

	f=BinaryFile("label_train_twoclass.bin","w")
	lab.save(f)

	f=HDF5File("label_train_real.h5","w", "/data/labels")
	lab.save(f)

	f=CSVFile("label_train_twoclass.ascii")
	lab2.load(f)

	f=BinaryFile("label_train_twoclass.bin")
	lab2.load(f)

	f=HDF5File("fm_train_real.h5","r", "/data/doubles")
	feats2.load(f)
	#print(feats2.get_feature_matrix())
	f=HDF5File("label_train_real.h5","r", "/data/labels")
	lab2.load(f)
	#print(lab2.get_labels())

	#clean up
	import os
	for f in ['fm_train_sparsereal.bin','fm_train_sparsereal.ascii',
			'fm_train_real.bin','fm_train_real.h5','fm_train_real.ascii',
			'label_train_real.h5', 'label_train_twoclass.ascii','label_train_twoclass.bin']:
		os.unlink(f)
	return feats, feats2, lab, lab2

if __name__=='__main__':
	print('Features IO')
	features_io_modular(*parameter_list[0])
