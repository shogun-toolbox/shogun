#!/usr/bin/env python
parameter_list=[[10,0.3,2, 1.0, 0.1]]

def check_status(status,suffix):
	# silent...
	assert status, "ERROR reading/writing status:%s/suffic:%s\n" % (status,suffix)

def serialization_complex_example (num=5, dist=1, dim=10, C=2.0, width=10):
	import os
	from numpy import concatenate, zeros, ones
	from numpy.random import randn, seed
	from modshogun import RealFeatures, MulticlassLabels
	from modshogun import GMNPSVM
	from modshogun import GaussianKernel
	from modshogun import SerializableHdf5File,SerializableAsciiFile, \
			SerializableJsonFile,SerializableXmlFile,MSG_DEBUG
	from modshogun import NormOne, LogPlusOne
	from tempfile import NamedTemporaryFile

	seed(17)

	data=concatenate((randn(dim, num), randn(dim, num) + dist,
					  randn(dim, num) + 2*dist,
					  randn(dim, num) + 3*dist), axis=1)
	lab=concatenate((zeros(num), ones(num), 2*ones(num), 3*ones(num)))

	feats=RealFeatures(data)
	#feats.io.set_loglevel(MSG_DEBUG)
	#feats.io.enable_file_and_line()
	kernel=GaussianKernel(feats, feats, width)

	labels=MulticlassLabels(lab)

	svm = GMNPSVM(C, kernel, labels)

	feats.add_preprocessor(NormOne())
	feats.add_preprocessor(LogPlusOne())
	feats.set_preprocessed(1)
	svm.train(feats)
	bias_ref = svm.get_svm(0).get_bias()

	#svm.print_serializable()

	tmp_h5 = NamedTemporaryFile(suffix='h5')
	fstream = SerializableHdf5File(tmp_h5.name, "w")
	status = svm.save_serializable(fstream)
	check_status(status,'h5')

	tmp_asc = NamedTemporaryFile(suffix='asc')
	fstream = SerializableAsciiFile(tmp_asc.name, "w")
	status = svm.save_serializable(fstream)
	check_status(status,'asc')

	tmp_json = NamedTemporaryFile(suffix='json')
	fstream = SerializableJsonFile(tmp_json.name, "w")
	status = svm.save_serializable(fstream)
	check_status(status,'json')

	tmp_xml = NamedTemporaryFile(suffix='xml')
	fstream = SerializableXmlFile(tmp_xml.name, "w")
	status = svm.save_serializable(fstream)
	check_status(status,'xml')

	fstream = SerializableHdf5File(tmp_h5.name, "r")
	new_svm=GMNPSVM()
	status = new_svm.load_serializable(fstream)
	check_status(status,'h5')
	new_svm.train()
	bias_h5 = new_svm.get_svm(0).get_bias()

	fstream = SerializableAsciiFile(tmp_asc.name, "r")
	new_svm=GMNPSVM()
	status = new_svm.load_serializable(fstream)
	check_status(status,'asc')
	new_svm.train()
	bias_asc = new_svm.get_svm(0).get_bias()

	fstream = SerializableJsonFile(tmp_json.name, "r")
	new_svm=GMNPSVM()
	status = new_svm.load_serializable(fstream)
	check_status(status,'json')
	new_svm.train()
	bias_json = new_svm.get_svm(0).get_bias()

	fstream = SerializableXmlFile(tmp_xml.name, "r")
	new_svm=GMNPSVM()
	status = new_svm.load_serializable(fstream)
	check_status(status,'xml')
	new_svm.train()
	bias_xml = new_svm.get_svm(0).get_bias()

	return svm,new_svm, bias_ref, bias_h5, bias_asc, bias_json, bias_xml


if __name__=='__main__':
	print('Serialization')
	serialization_complex_example(*parameter_list[0])
