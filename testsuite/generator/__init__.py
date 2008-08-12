"""A package to generate testcases for the Shogun toolbox

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation in version 3.

Written (W) 2007-2008 Sebastian Henschel
Copyright (C) 2007-2008 Fraunhofer Institute FIRST and Max-Planck-Society
"""

import sys
import os
import numpy.random as random

from shogun.Library import Math_init_random
from dataop import INIT_RANDOM
from fileop import clean_dir_outdata

MODULES=['classifier', 'clustering', 'distance', 'distribution', 'kernel', \
	'regression', 'preproc']

def run (argv):
	"""
	Run all individual generators or only one if present in
	argument list.
	"""

	# put some constantness into randomness
	Math_init_random(INIT_RANDOM)
	random.seed(INIT_RANDOM)

	arglen=len(argv)
	if arglen==2: # run given module
		if argv[1]=='clear':
			clean_dir_outdata()
		else:
			__import__(argv[1], globals(), locals());
			module=eval(argv[1])
			module.run()
	else:
		# run given modules by calling self again, one by one
		# this is due to an issue somewhere with classifiers (atm) and
		# 'static randomness'

		if arglen==1:
			command=argv[0]
			mods=MODULES
		else:
			command=argv.pop(0)
			mods=argv

		for mod in mods:
			if not mod in MODULES:
				mods=', '.join(MODULES)
				msg="Unknown module: %s\nTry one of these: %s\n"%(mod, mods)
				sys.stderr.write(msg)
				sys.exit(1)

			ret=os.system('%s %s' % (command, mod))
			if ret!=0:
				sys.exit(ret)

