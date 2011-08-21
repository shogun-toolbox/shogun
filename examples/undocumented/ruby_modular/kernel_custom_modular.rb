# this was trancekoded by the awesome trancekoder
# ...and fixifikated by the awesum fixifikator
require 'modshogun'
require 'pp'
seed(42)

parameter_list=[[7],[8]]

def kernel_custom_modular(dim=7)

	seed(17)
	data=rand(dim, dim)
# *** 	feats=RealFeatures(data)
	feats=Modshogun::RealFeatures.new
	feats.set_features(data)
	symdata=data+data.T
	lowertriangle=array([symdata[(x,y)] for x in xrange(symdata.shape[1])
		for y in xrange(symdata.shape[0]) if y<=x])

# *** 	kernel=CustomKernel()
	kernel=Modshogun::CustomKernel.new
	kernel.set_features()

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
	return km_fullfull,kernel


end
if __FILE__ == $0
	puts 'Custom'
	kernel_custom_modular(*parameter_list[0])


end
