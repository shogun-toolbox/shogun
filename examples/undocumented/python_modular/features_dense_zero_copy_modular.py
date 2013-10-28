#!/usr/bin/env python
import numpy
from modshogun import RealFeatures
from numpy import array, float64, int64

# create dense matrice
data=[[1,2,3],[4,5,6],[7,8,9],[-1,-2,-3]]

parameter_list = [[data]]

def features_dense_zero_copy_modular (in_data=data):
	feats = None
	if numpy.__version__ >= '1.5':
		feats=numpy.array(in_data, dtype=float64, order='F')

		a=RealFeatures()
		a.frombuffer(feats, False)

		b=numpy.array(a, copy=False)
		c=numpy.array(a, copy=True)

		d=RealFeatures()
		d.frombuffer(a, False)

		e=RealFeatures()
		e.frombuffer(a, True)

		a[:,0]=0
		#print a[0:4]
		#print b[0:4]
		#print c[0:4]
		#print d[0:4]
		#print e[0:4]
	else:
		print("numpy version >= 1.5 is needed")

	return feats

if __name__=='__main__':
	print('dense_zero_copy')
	features_dense_zero_copy_modular(*parameter_list[0])
