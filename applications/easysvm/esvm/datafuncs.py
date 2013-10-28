"""
This module contains code for generating toy examples
"""

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
import parse

import random
from numpy.random import randn
from numpy import ones, concatenate, array, transpose
from esvm.mldata import DatasetFileFASTA, init_datasetfile
from esvm.mldata_arff import DatasetFileARFF

class MotifDataDef(object):
    motif = ''
    numseq = 0
    seqlenmin = 0
    seqlenmax = 0
    posstart = 0
    posend = 0
    mutrate = 0.0

################################################################################
# data generation functions

def motifgen(motif, numseq, seqlenmin, seqlenmax, posstart, posend, mutrate):
    """Generate sequences with a particular motif at a particular location.
    Also allow a possible mutation rate of the motif.
    """

    metadata = 'motifgen(%s,%d,%d,%d,%d,%d,%1.2f)' % (motif, numseq, seqlenmin, seqlenmax, posstart, posend, mutrate)

    acgt='acgt'
    seqlist = []
    for i in xrange(0,numseq):
        str=[] ;
        seqlen=random.randint(seqlenmin,seqlenmax) ;
        for l in xrange(0,seqlen):
            str.append(acgt[random.randint(0,3)])
        pos=random.randint(posstart,posend) ;
        for l in xrange(0,len(motif)):
            if (random.random()>=mutrate) and (pos+l<seqlen) and (pos+l>=0):
                str[pos+l]=motif[l]
        seqlist.append(''.join(str).upper())

    return metadata, seqlist


def cloudgen(numpoint, numfeat, fracpos, width):
    """Generate two Gaussian point clouds, centered around one and minus one."""

    numpos = int(round(fracpos*numpoint))
    numneg = numpoint - numpos

    metadata = 'cloudgen(%d,%d,%d,%3.2f)' % (numpos, numneg, numfeat, width)

    datapos = ones((numfeat, numpos)) + width*randn(numfeat, numpos)
    dataneg = -ones((numfeat, numneg)) + width*randn(numfeat, numneg)
    pointcloud = concatenate((datapos,dataneg),axis=1)
    labels = concatenate((ones(numpos),-ones(numneg)))

    return metadata, pointcloud, labels





################################################################################
# ARFF functions

def arffwrite_real(filename, numpoint, numfeat, fracpos=0.5, width=1.0):
    """Write an ARFF file containing a vectorial dataset"""
    #import arff

    (metadata, pointcloud, labels) = cloudgen(numpoint, numfeat, fracpos, width)

    fp = init_datasetfile(filename,'vec')
    fp.comment = metadata
    fp.dataname = 'pointcloud'
    fp.writelines(pointcloud,labels)


def arffwrite_sequence(filename,p, n):
    """Write an ARFF file containing a sequence dataset"""
    #import arff

    (metadatapos,seqlistpos) = motifgen(p.motif, p.numseq, p.seqlenmin, p.seqlenmax, p.posstart, p.posend, p.mutrate)
    (metadataneg,seqlistneg) = motifgen(n.motif, n.numseq, n.seqlenmin, n.seqlenmax, n.posstart, n.posend, n.mutrate)

    labels = concatenate((ones(len(seqlistpos)),-ones(len(seqlistneg))))
    seqlist = seqlistpos + seqlistneg
    fp = init_datasetfile(filename,'seq')
    fp.comment = metadatapos+' '+metadataneg
    fp.dataname = 'motif'
    fp.writelines(seqlist,labels)



def arffread(kernelname,datafilename):
    """Decide based on kernelname whether to read a sequence or vectorial file"""

    if kernelname == 'gauss' or kernelname == 'linear' or kernelname == 'poly' or kernelname == None:
        fp = init_datasetfile(datafilename,'vec')
    elif kernelname == 'wd' or kernelname == 'localalign' or kernelname == 'localimprove'\
             or kernelname == 'spec' or kernelname == 'cumspec':
        fp = init_datasetfile(datafilename,'seq')
    elif kernelname == 'spec2' or kernelname == 'cumspec2':
        fp = init_datasetfile(datafilename,'mseq')
    else:
        print 'Unknown kernel in arffread'

    return fp.readlines()

################################################################################
# fasta functions

def fastawrite_sequence(filename,p):
    """Write a FASTA file containing a sequence dataset"""
    import arff

    (metadata,seqlist) = motifgen(p.motif, p.numseq, p.seqlenmin, p.seqlenmax, p.posstart, p.posend, p.mutrate)
    labels = ones(len(seqlist))
    fp = init_datasetfile(filename,'seq')
    fp.writelines(seqlist,labels)

def fastaread(fnamepos,fnameneg=None):
    """Read two fasta files, the first positive, the second negative"""
    fpos = init_datasetfile(fnamepos,'seq')
    (fa1,lab1) = fpos.readlines()

    if fnameneg is not None:
        fneg = init_datasetfile(fnameneg,'seq')
        (fa2,lab2) = fneg.readlines()

        print 'positive: %d, negative %d' % (len(fa1),len(fa2))
        all_labels = concatenate((ones(len(fa1)),-ones(len(fa2))))
        all_examples = fa1 + fa2
    else:
        all_examples = fa1
        all_labels = ones(len(fa1))

    return all_examples, all_labels

