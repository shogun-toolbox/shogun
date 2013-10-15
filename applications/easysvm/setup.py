#!/usr/bin/env python

#############################################################################################
#                                                                                           #
#    This program is free software; you can redistribute it and/or modify                   #
#    it under the terms of the GNU General Public License as published by                   #
#    the Free Software Foundation; either version 3 of the License, or                      #
#    (at your option) any later version.                                                    #
#                                                                                           #
#    This program is distributed in the hope that it will be useful,                        #
#    but WITHOUT ANY WARRANTY; without even the implied warranty of                         #
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the                           #
#    GNU General Public License for more details.                                           #
#                                                                                           #
#    You should have received a copy of the GNU General Public License                      #
#    along with this program; if not, see http://www.gnu.org/licenses                       #
#    or write to the Free Software Foundation, Inc., 51 Franklin Street,                    #
#    Fifth Floor, Boston, MA 02110-1301  USA                                                #
#                                                                                           #
#############################################################################################

import sys
from distutils.core import setup, Extension


sys.stdout.write('-------------------------------------------\n')
sys.stdout.write('easysvm - A front end to the shogun toolbox\n')
sys.stdout.write('-------------------------------------------\n\n')


try:
    import numpy
except:
    sys.stderr.write("WARNING: did not find 'numpy'\n")

try:
    import pylab
except:
    sys.stderr.write("WARNING: did not find 'pylab'\n")

try:
    import shogun
except:
    sys.stderr.write("WARNING: did not find 'shogun'\n")
    sys.stderr.write("         shogun is a required back end.\n")
    sys.stderr.write("         See shogun website: http://www.shogun-toolbox.org)\n")

try:
    import arff
except:
    sys.stderr.write("WARNING: did not find 'arff'\n")
    sys.stderr.write("         arff is required for reading and writing ARFF data files\n")
    sys.stderr.write("         See arff website: http://www.mit.edu/~sav/arff/\n\n")
    sys.stderr.write("         All other functionality should be ok.\n\n")



setup (name = 'easysvm',
       version = '0.3.3',
       description = 'easysvm - A front end to the shogun toolbox',
       author = ['Cheng Soon Ong', 'Gunnar Raetsch' ],
       author_email = ['chengsoon.ong@tuebingen.mpg.de','gunnar.raetsch@tuebingen.mpg.de'],
       license='GPLv3',
       url = 'http://www.fml.tuebingen.mpg.de/raetsch/projects/easysvm',
       py_modules=['esvm.datafuncs','esvm.mldata','esvm.mldata_arff','esvm.experiment','esvm.parse',\
                   'esvm.plots','esvm.poim','esvm.utils','splicesites.utils'],
       packages=['esvm','splicesites'],
       scripts=['scripts/easysvm.py','scripts/datagen.py'],
       long_description="""
       easysvm is a front end to the shogun toolbox. It aims to be a
       'quick start' tutorial for users interested in shogun. As such,
       it covers only the absolute basics of machine learning.
       More advanced users should directly use the interfaces provided
       by shogun.

       This setup.py installs two modules:
       - esvm (which contains some basic tools for machine learning with SVMs)
       - splicesites (which contains tools for splice site prediction)

       It also installs two scripts (easysvm.py and datagen.py) in the relevant binary directory.
       The two scripts provide a command line interface to the toolkit.
       """
       )

