"""generator

A package to generate testcases for the Shogun toolbox.

"""

__license__='GPL v2'
__url__='http://shogun-toolbox.org'

from numpy.random import seed

from fileops import clean_dir_output
import kernels
import distances


def run ():
	#seed(None)
	seed(42)
# looks like the value of random's state depends on what kernels
# are executed when and in which order. so everytime a new kernel is added,
# for instance, the generated testdata will be different.
#-565350764
#-952662014
#-649480086
#   import numpy
#	print sum(numpy.random.get_state()[1])

	clean_dir_output()
	kernels.run()
	distances.run()


