#!/usr/bin/env python

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# Written (W) 2008-2009 Soeren Sonnenburg
# Copyright (C) 2008-2009 Fraunhofer Institute FIRST and Max-Planck-Society

import numpy
import os
import sys
import optparse
import bz2
from signal_sensor import SignalSensor
from genomic import read_single_fasta

arts_version = 'v0.3'
def_file = bz2.BZ2File('data/ARTS.dat.bz2')

def print_version():
    sys.stderr.write('arts ' + arts_version + '\n')

def parse_options():
    parser = optparse.OptionParser(usage="usage: %prog [options] seq.fa")

    parser.add_option("-o", "--outfile", type="str", default='stdout',
                              help="File to write the results to")
    parser.add_option("-v", "--version", default=False,
                              help="Show some more information")
    parser.add_option("--organism", type="str", default='Worm',
                              help="""use model for organism when predicting
                              (one of Cress, Fish, Fly, Human, Worm)""")

    (options, args) = parser.parse_args()
    if options.version:
        print_version()
        sys.exit(0)

    if len(args) != 1:
        parser.error("incorrect number of arguments")

    fafname = args[0]
    if not os.path.isfile(fafname):
        parser.error("fasta file does not exist")

    if options.outfile == 'stdout':
        outfile = sys.stdout
    else:
        try:
            outfile = file(options.outfile, 'w')
        except IOError:
            parser.error("could not open %s for writing" % options.outfile)

    return (fafname, outfile)

if __name__ == '__main__':
    (fafname, outfile) = parse_options()
    seq = read_single_fasta(fafname)

    arts = SignalSensor()
    arts.from_file(def_file)
    preds = arts.predict(seq)

    for p in preds:
        outfile.write('%+g\n' % p)
