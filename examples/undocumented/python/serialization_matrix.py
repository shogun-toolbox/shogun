#!/usr/bin/env python
from modshogun import *
from numpy import array

parameter_list=[[[[1.0,2,3],[4,5,6]]]]

def serialization_matrix (m):
	from tempfile import NamedTemporaryFile
	feats=RealFeatures(array(m))
	#feats.io.set_loglevel(0)
	tmp_asc_1 = NamedTemporaryFile(suffix='1.asc')
	fstream = SerializableAsciiFile(tmp_asc_1.name, "w")
	feats.save_serializable(fstream)

	tmp_asc_2 = NamedTemporaryFile(suffix='2.asc')
	l=MulticlassLabels(array([1.0,2,3]))
	fstream = SerializableAsciiFile(tmp_asc_2.name, "w")
	l.save_serializable(fstream)

if __name__=='__main__':
	print('Serialization Matrix Modular')
	serialization_matrix(*parameter_list[0])
