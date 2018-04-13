#!/usr/bin/env python

# This software is distributed under BSD 3-clause license (see LICENSE file).
#
# Authors: Soeren Sonnenburg

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

