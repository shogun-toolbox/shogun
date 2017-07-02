#!/usr/bin/env python
from tools.load import LoadMatrix
lm=LoadMatrix()
data=lm.load_numbers('../data/fm_train_real.dat')
label=lm.load_numbers('../data/label_train_twoclass.dat')

parameter_list=[[data,label]]

def features_io (fm_train_real, label_train_twoclass):
	import numpy
	from modshogun import SparseRealFeatures, RealFeatures, MulticlassLabels
	from modshogun import GaussianKernel
	from modshogun import LibSVMFile, CSVFile, BinaryFile, HDF5File
	from tempfile import NamedTemporaryFile

	feats=SparseRealFeatures(fm_train_real)
	feats2=SparseRealFeatures()

	tmp_fm_train_sparsereal_bin = NamedTemporaryFile(suffix='sparsereal.bin')
	f=BinaryFile(tmp_fm_train_sparsereal_bin.name, "w")
	feats.save(f)

	tmp_fm_train_sparsereal_ascii = NamedTemporaryFile(suffix='sparsereal.ascii')
	f=LibSVMFile(tmp_fm_train_sparsereal_ascii.name, "w")
	feats.save(f)

	f=BinaryFile(tmp_fm_train_sparsereal_bin.name)
	feats2.load(f)

	f=LibSVMFile(tmp_fm_train_sparsereal_ascii.name)
	feats2.load(f)

	feats=RealFeatures(fm_train_real)
	feats2=RealFeatures()

	tmp_fm_train_real_bin = NamedTemporaryFile(suffix='real.bin')
	f=BinaryFile(tmp_fm_train_real_bin.name, "w")
	feats.save(f)

	tmp_fm_train_real_h5 = NamedTemporaryFile(suffix='real.h5')
	f=HDF5File(tmp_fm_train_real_h5.name, "w", "/data/doubles")
	feats.save(f)

	tmp_fm_train_real_ascii = NamedTemporaryFile(suffix='real.ascii')
	f=CSVFile(tmp_fm_train_real_ascii.name, "w")
	feats.save(f)

	f=BinaryFile(tmp_fm_train_real_bin.name)
	feats2.load(f)
	#print("diff binary", numpy.max(numpy.abs(feats2.get_feature_matrix().flatten()-fm_train_real.flatten())))

	f=CSVFile(tmp_fm_train_real_ascii.name)
	feats2.load(f)
	#print("diff ascii", numpy.max(numpy.abs(feats2.get_feature_matrix().flatten()-fm_train_real.flatten())))

	lab=MulticlassLabels(numpy.array([0.0,1.0,2.0,3.0]))
	lab2=MulticlassLabels()
	tmp_label_train_twoclass_ascii = NamedTemporaryFile(suffix='twoclass.ascii')
	f=CSVFile(tmp_label_train_twoclass_ascii.name, "w")
	lab.save(f)

	tmp_label_train_twoclass_bin = NamedTemporaryFile(suffix='twoclass.bin')
	f=BinaryFile(tmp_label_train_twoclass_bin.name, "w")
	lab.save(f)

	tmp_label_train_real_h5 = NamedTemporaryFile(suffix='real.h5')
	f=HDF5File(tmp_label_train_real_h5.name, "w", "/data/labels")
	lab.save(f)

	f=CSVFile(tmp_label_train_twoclass_ascii.name)
	lab2.load(f)

	f=BinaryFile(tmp_label_train_twoclass_bin.name)
	lab2.load(f)

	f=HDF5File(tmp_fm_train_real_h5.name, "r", "/data/doubles")
	feats2.load(f)
	#print(feats2.get_feature_matrix())
	f=HDF5File(tmp_label_train_real_h5.name, "r", "/data/labels")
	lab2.load(f)
	#print(lab2.get_labels())

	return feats, feats2, lab, lab2

if __name__=='__main__':
	print('Features IO')
	features_io(*parameter_list[0])
