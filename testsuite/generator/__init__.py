"""generator

A package to generate testcases for the shogun toolbox.

"""

__license__='GPL v2'
__url__='http://shogun-toolbox.org'

from numpy.random import seed

import fileops
import kernels
import distance


def run ():
	#seed(None)
	seed(42)

	fileops.clean_dir_output()
	kernels.run()
	#distance.run()


