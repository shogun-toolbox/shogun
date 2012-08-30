#!/usr/bin/env python
from modshogun import *
from numpy import array
import os

parameter_list=[[[[1.0,2,3],[4,5,6]]]]

def serialization_matrix_modular (m):
	feats=RealFeatures(array(m))
	#feats.io.set_loglevel(0)
	fstream = SerializableAsciiFile("foo.asc", "w")
	feats.save_serializable(fstream)

	l=MulticlassLabels(array([1.0,2,3]))
	fstream = SerializableAsciiFile("foo2.asc", "w")
	l.save_serializable(fstream)

	os.unlink("foo.asc")
	os.unlink("foo2.asc")

if __name__=='__main__':
	print('Serialization Matrix Modular')
	serialization_matrix_modular(*parameter_list[0])
