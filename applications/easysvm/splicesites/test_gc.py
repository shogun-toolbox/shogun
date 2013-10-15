#!/usr/bin/env python

import bz2
import time
import sys
import numpy
import numpy.matlib
from splicesites.utils import create_dataset
from esvm.utils import calcroc
from esvm.experiment import crossvalidation
from esvm.mldata import init_datasetfile

def test_gc(gcfilename):
    """
    Check the gc content files for conflicting labels
    """
    fp = init_datasetfile(gcfilename,'vec')
    (examples,labels) = fp.readlines()
    print '%d positive and %d negative examples' % (sum(labels>0.0),sum(labels<0.0))

    distance = sqr_dist(numpy.matrix(examples),numpy.matrix(examples))
    labdist = numpy.matrix(labels).T*numpy.matrix(labels)
    #difflab = numpy.where(labdist.A<0,distance,numpy.matlib.ones((len(labels),len(labels))))
    contracount = 0
    for ix in xrange(len(labels)):
        for iy in xrange(ix+1,len(labels)):
            if labdist[ix,iy]<0 and distance[ix,iy]<0.01:
                contracount += 1
    print distance.shape, labdist.shape
    #print '%d identical examples with opposing labels' %len(numpy.unique(numpy.where(difflab==0)[0]))
    print '%d identical examples with opposing labels' % contracount


def sqr_dist(a,b):
    """Compute the square distance between vectors"""
    dot_a = numpy.sum(numpy.multiply(a,a),axis=0).T
    dot_b = numpy.sum(numpy.multiply(b,b),axis=0).T
    unitvec = numpy.matlib.ones(dot_a.shape)
    D = 2.0*a.T*b

    for ix,bval in enumerate(dot_b):
        D[:,ix] = dot_a - D[:,ix] + numpy.kron(bval,unitvec)

    return D


if __name__ == '__main__':
    test_gc('C_elegans_don_freq.csv')
    test_gc('C_elegans_acc_freq.csv')

