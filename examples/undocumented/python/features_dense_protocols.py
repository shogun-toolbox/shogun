#!/usr/bin/env python
import numpy
from modshogun import RealFeatures
from modshogun import LongIntFeatures

from numpy import array, float64, int64

# create dense matrice
data=[[1,2,3],[4,5,6],[7,8,9],[-1,-2,-3]]

parameter_list = [[data]]

def features_dense_protocols_modular (in_data=data):
	m_real=array(in_data, dtype=float64, order='F')
	f_real=RealFeatures(m_real)

	#print m_real
	#print f_real

	#print f_real[-1]
	#print f_real[1, 2]
	#print f_real[-1:3]
	#print f_real[2, 0:2]
	#print f_real[0:3, 1]
	#print f_real[0:3, 1:2]
	#print f_real[:,1]
	#print f_real[1,:]

	#print m_real[-2]
	f_real[-1]=m_real[-2]
	#print f_real[-1]

	#print m_real[0, 1]
	f_real[1,2]=m_real[0,1]
	#print f_real[1, 2]

	#print m_real[0:2]
	f_real[1:3]=m_real[0:2]
	#print f_real[1:3]

	#print m_real[0, 0:2]
	f_real[2, 0:2]=m_real[0,0:2]
	#print f_real[2, 0:2]

	#print m_real[0:3, 2]
	f_real[0:3,1]=m_real[0:3, 2]
	#print f_real[0:3, 1]

	#print m_real[0:3, 0:1]
	f_real[0:3,1:2]=m_real[0:3,0:1]
	#print f_real[0:3, 1:2]

	f_real[:,0]=0
	#print f_real.get_feature_matrix()

	if numpy.__version__ >= '1.5':
		f_real+=m_real
		f_real*=m_real
		f_real-=m_real
	else:
		print("numpy version >= 1.5 is needed")
		return None

	f_real+=f_real
	f_real*=f_real
	f_real-=f_real

	#print f_real
	#print f_real.get_feature_matrix()

	try:
		mem_real=memoryview(f_real)
	except NameError:
		print("Python2.7 and later is needed for memoryview class")
		return

	ret_real=array(f_real)
	#print ret_real

	return f_real[:,0]

if __name__=='__main__':
	print('dense_protocols')
	features_dense_protocols_modular(*parameter_list[0])
