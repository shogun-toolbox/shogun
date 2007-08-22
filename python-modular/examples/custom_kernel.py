from numpy.random import rand
from shogun.Kernel import CustomKernel

k=CustomKernel()
k.set_full_kernel_matrix_from_full(rand(10,10))
