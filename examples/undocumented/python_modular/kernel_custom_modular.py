#!/usr/bin/env python
from numpy.random import seed
seed(42)

parameter_list=[[7],[8]]

def kernel_custom_modular (dim=7):
	from numpy.random import rand, seed
	from numpy import array, float32
	from modshogun import RealFeatures
	from modshogun import CustomKernel

	seed(17)
	data=rand(dim, dim)
	feats=RealFeatures(data)
	symdata=data+data.T
	lowertriangle=array([symdata[(x,y)] for x in range(symdata.shape[1])
		for y in range(symdata.shape[0]) if y<=x])

	kernel=CustomKernel()

	# once with float64's
	kernel.set_triangle_kernel_matrix_from_triangle(lowertriangle)
	km_triangletriangle=kernel.get_kernel_matrix()

	kernel.set_triangle_kernel_matrix_from_full(symdata)
	km_fulltriangle=kernel.get_kernel_matrix()

	kernel.set_full_kernel_matrix_from_full(symdata)
	km_fullfull=kernel.get_kernel_matrix()

	# now once with float32's
	data=array(data,dtype=float32)

	kernel.set_triangle_kernel_matrix_from_triangle(lowertriangle)
	km_triangletriangle=kernel.get_kernel_matrix()

	kernel.set_triangle_kernel_matrix_from_full(symdata)
	km_fulltriangle=kernel.get_kernel_matrix()

	kernel.set_full_kernel_matrix_from_full(symdata)
	km_fullfull=kernel.get_kernel_matrix()
	return km_fullfull,kernel

if __name__=='__main__':
	print('Custom')
	kernel_custom_modular(*parameter_list[0])

