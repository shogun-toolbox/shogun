"""A package to generate testcases for the Shogun toolbox

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation in version 3.

Written (W) 2007-2008 Sebastian Henschel
Copyright (C) 2007-2008 Fraunhofer Institute FIRST and Max-Planck-Society
"""

from fileop import clean_dir_outdata
import classifier
import clustering
import distance
import distribution
import kernel
import regression
import preproc


def run ():
	"""Run all the individual generators."""

	clean_dir_outdata()
	kernel.run()
	distance.run()
	classifier.run()
	clustering.run()
	distribution.run()
	regression.run()
	preproc.run()
