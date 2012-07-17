import numpy
from shogun.Features import RealFeatures
from shogun.Features import LongIntFeatures

from numpy import array, float64, int64

# create dense matrice
data=[[1,2,3],[4,0,0],[0,0,0],[0,5,0],[0,0,6],[9,9,9]]

parameter_list = [[data]]

def features_dense_real_modular(in_data=data):
	m_real=array(in_data, dtype=float64, order='F')
	m_long=array(in_data, dtype=int64, order='F')

	f_real=RealFeatures(m_real)
	f_long=LongIntFeatures(m_long)

	if numpy.__version__ >= 1.5:
		f_real+=m_real
		f_long+=m_long

		f_real*=m_real
		f_long*=m_long

		f_real-=m_real
		f_long-=m_long
	else:
		print "numpy version >= 1.5 is needed"

	f_real+=f_real
	f_long+=f_long

	f_real*=f_real
	f_long*=f_long

	f_real-=f_real
	f_long-=f_long

#	print f_real
#	print f_long

	try:
		mem_real=memoryview(f_real)
		mem_long=memoryview(f_long)
	except NameError:
#		print "Python2.7 is needed for memoryview class"
		pass

	ret_real=array(f_real)
	ret_long=array(f_long)

	print ret_real
	print ret_long

if __name__=='__main__':
	print('dense_real')
	features_dense_real_modular(*parameter_list[0])
