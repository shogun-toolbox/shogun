#!/usr/bin/env python

# This software is distributed under BSD 3-clause license (see LICENSE file).
#
# Authors: Soeren Sonnenburg

import sys
import random
from numpy import array
import esvm.parse
import esvm.plots
from esvm.datafuncs import MotifDataDef, fastawrite_sequence, arffwrite_sequence, arffwrite_real
from esvm.mldata import init_datasetfile

if __name__ == '__main__':

    if len(sys.argv)<3 or (sys.argv[1]=='motif' and sys.argv[2]!='arff' and sys.argv[2]!='fasta') \
           or (sys.argv[1]=='motif' and sys.argv[2]=='fasta' and len(sys.argv)<9) \
           or (sys.argv[1]=='motif' and sys.argv[2]=='arff' and len(sys.argv)<14) \
           or (sys.argv[1]=='cloud' and len(sys.argv)<7) or (sys.argv[1]!='motif') \
           and (sys.argv[1]!='cloud'):
        sys.stderr.write( "usage: %s motif fasta MOTIF numSeq seqLenRange"+\
                          "positionRange mutationRate output.fa\n"+\
                          "or: %s motif arff MOTIFPOS numSeq-pos seqLenRange-pos "+\
                          "positionRange-pos mutationRate-pos \\\n"+\
                          "motif-neg numSeq-neg seqLenRange-neg positionRange-neg "+\
                          "mutationRange-neg output.arff\n"+\
                          "or: %s cloud numpoints dimensions fractionOfPositives "+\
                          "cloudWidth output.arff\n" % (sys.argv[0],sys.argv[0],sys.argv[0]) )
        sys.exit(-1)

    random.seed()

    if sys.argv[1] == 'motif':
        if sys.argv[2]=='fasta':
            # generate sequences in FASTA format
            p = MotifDataDef()
            p.motif = sys.argv[3]
            p.numseq = int(sys.argv[4])
            (p.seqlenmin,p.seqlenmax) = esvm.parse.parse_range(sys.argv[5])
            (p.posstart,p.posend) = esvm.parse.parse_range(sys.argv[6])
            p.mutrate = float(sys.argv[7])

            filename = sys.argv[8]
            fastawrite_sequence(filename, p)

        else:
            # generate sequences in ARFF format
            assert(sys.argv[2]=='arff')
            p = MotifDataDef()
            p.motif = sys.argv[3]
            p.numseq = int(sys.argv[4])
            (p.seqlenmin,p.seqlenmax) = esvm.parse.parse_range(sys.argv[5])
            (p.posstart,p.posend) = esvm.parse.parse_range(sys.argv[6])
            p.mutrate = float(sys.argv[7])

            n = MotifDataDef()
            n.motif = sys.argv[8]
            n.numseq = int(sys.argv[9])
            (n.seqlenmin,n.seqlenmax) = esvm.parse.parse_range(sys.argv[10])
            (n.posstart,n.posend) = esvm.parse.parse_range(sys.argv[11])
            n.mutrate = float(sys.argv[12])

            filename = sys.argv[13]
            arffwrite_sequence(filename, p, n)

    elif sys.argv[1] == 'cloud':
        # generate a data cloud in ARFF format
        numpoint = int(sys.argv[2])
        numfeat = int(sys.argv[3])
        fracpos = float(sys.argv[4])
        width = float(sys.argv[5])

        filename = sys.argv[6]
        arffwrite_real(filename, numpoint, numfeat, fracpos, width)
        if len(sys.argv)>=8:
            fp = init_datasetfile(filename,'vec')
            (examples,labels) = fp.readlines()
            pointcloud = []
            for ix in xrange(numpoint):
                pointcloud.append(array([labels[ix],examples[0,ix],examples[1,ix]]))
            esvm.plots.plotcloud(pointcloud,sys.argv[7],'Pointcloud')

	#(examples,labels,metadata)=arffwrite_real(filename, numpoint, numfeat, fracpos, width)
	#if len(sys.argv)>=8:
	#	plots.plotcloud(pointcloud,sys.argv[7],metadata)
    else:
        print 'Unknown option %s\n' % sys.argv[1]
