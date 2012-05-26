import math
from shogun.Library import DynamicIntArray

class AbsDynamicIntArray(DynamicIntArray):
	def __init__(self, dim1_size, dim2_size=1, dim3_size=1):
		DynamicIntArray.__init__(self, dim1_size, dim2_size, dim3_size)
	def dset_element(self, e, idx1, idx2=0, idx3=0):
		self.set_element(abs(e), idx1, idx2, idx3)

def director_dynamicarray_modular():

	dyna1=DynamicIntArray(2)
	dyna2=AbsDynamicIntArray(2)

	dyna1.dset_element(5, 0)
	dyna1.dset_element(-5, 1)

	dyna2.dset_element(5, 0)
	dyna2.dset_element(-5, 1)

	print dyna1.get_element(0), dyna1.get_element(1)
	print dyna2.get_element(0), dyna2.get_element(1)

if __name__=='__main__':
	print('DirectorDynamicArray')
	director_dynamicarray_modular()
