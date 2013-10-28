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
import random
from esvm.experiment import svm_cv, svm_pred, svm_poim, svm_eval, svm_modelsel

if __name__ == '__main__':

    if len(sys.argv)<2:
        sys.stderr.write("usage: %s [cv|pred|modelsel|eval|poim] parameters\n" % sys.argv[0])
        sys.exit(-1)

    random.seed()

    topmode = sys.argv[1]

    if topmode == 'cv':
        svm_cv(sys.argv)
    elif topmode == 'pred':
        svm_pred(sys.argv)
    elif topmode == 'poim':
        svm_poim(sys.argv)
    elif topmode == 'eval':
        svm_eval(sys.argv)
    elif topmode == 'modelsel':
        svm_modelsel(sys.argv)
    else:
        sys.stderr.write( "unknown mode %s (use: cv, pred, poim, eval)\n" % topmode)
        sys.exit(-1)

    sys.exit(0)

