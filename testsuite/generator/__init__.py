"""generator

A package to generate testcases for the Shogun toolbox.

"""

__license__='GPL v2'

from numpy.random import seed

from fileop import clean_dir_outdata
import classifier
import clustering
import distance
import distribution
import kernel
import regression
import preproc


def run ():
	clean_dir_outdata()
	kernel.run()
	distance.run()
	classifier.run()
	clustering.run()
	distribution.run()
	regression.run()
	preproc.run()
