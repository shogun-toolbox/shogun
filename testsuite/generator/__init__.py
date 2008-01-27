"""A package to generate testcases for the Shogun toolbox

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation in version 3.

Written (W) 2007-2008 Sebastian Henschel
Copyright (C) 2007-2008 Fraunhofer Institute FIRST and Max-Planck-Society
"""

import sys
from fileop import clean_dir_outdata
MODULES=['classifier', 'clustering', 'distance', 'distribution', 'kernel', \
	'regression', 'preproc']

def run (argv):
	"""
	Run all individual generators or only one if present in
	argument list.
	"""

	clean_dir_outdata()

	if len(argv)<2:
		for mod in MODULES:
			__import__(mod, globals(), locals());
			module=eval(mod)
			module.run()
	else:
		argv=argv[1:]
		for mod in argv:
			try:
				__import__(mod, globals(), locals());
				module=eval(mod)
				module.run()
			except ImportError:
				mods=', '.join(MODULES)
				msg="Unknown module: %s\nTry one of these: %s\n"%(mod, mods)
				sys.stderr.write(msg)
				sys.exit(1)

