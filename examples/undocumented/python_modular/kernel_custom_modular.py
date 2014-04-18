#!/usr/bin/env python
from numpy.random import seed
seed(42)

parameter_list=[[7],[8]]

def kernel_custom_modular (dim=7):
	from numpy.random import rand, seed
	from numpy import array, float32, int32
	from modshogun import RealFeatures
	from modshogun import CustomKernel
	from modshogun import IndexFeatures

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

	# get subset of kernel
	row_idx=array(range(3),dtype=int32)
	col_idx=array(range(2),dtype=int32)
	row_idx_feat=IndexFeatures(row_idx)
	col_idx_feat=IndexFeatures(col_idx)
	kernel.init(row_idx_feat, col_idx_feat)
	km_sub_kernel=kernel.get_kernel_matrix()
	# print('Subkernel(3x2):\n%s'%km_sub_kernel)
	kernel.remove_all_col_subsets()
	kernel.remove_all_row_subsets()

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

