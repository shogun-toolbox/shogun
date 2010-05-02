def custom ():
	print 'Custom'
	from numpy.random import rand
	from numpy import array, float32
	from shogun.Features import RealFeatures
	from shogun.Kernel import CustomKernel

	dim=7
	data=rand(dim, dim)
	feats=RealFeatures(data)
	symdata=data+data.T
	lowertriangle=array([symdata[(x,y)] for x in xrange(symdata.shape[1])
		for y in xrange(symdata.shape[0]) if y<=x])

	kernel=CustomKernel()

	# once with float64's
	kernel.set_triangle_kernel_matrix_from_triangle(lowertriangle)
	km_triangletriangle=kernel.get_kernel_matrix()

	kernel.set_triangle_kernel_matrix_from_full(symdata)
	km_fulltriangle=kernel.get_kernel_matrix()

	kernel.set_full_kernel_matrix_from_full(data)
	km_fullfull=kernel.get_kernel_matrix()

	# now once with float32's
	data=array(data,dtype=float32)

	kernel.set_triangle_kernel_matrix_from_triangle(lowertriangle)
	km_triangletriangle=kernel.get_kernel_matrix()

	kernel.set_triangle_kernel_matrix_from_full(symdata)
	km_fulltriangle=kernel.get_kernel_matrix()

	kernel.set_full_kernel_matrix_from_full(data)
	km_fullfull=kernel.get_kernel_matrix()

if __name__=='__main__':
	from numpy.random import seed
	seed(42)
	custom()

