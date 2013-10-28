#!/usr/bin/env python

"""Classes to encapsulate the idea of a dataset in machine learning,
   including file access.

   This file contains the ARFF class for people who have arff installed.
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

try:
    import arff
    have_arff = True
except ImportError:
    have_arff = False


import sys
from numpy import array, concatenate
import csv
from esvm.mldata import DatasetFileBase

class DatasetFileARFF(DatasetFileBase):
    """Attribute-Relation File Format file, uses module arff.

    Labels are in the first column.
    """
    def __init__(self,filename,extype,dataname='ARFFdata',comment=''):
        """Do the base class init, then add some arff specific metadata"""
        if not have_arff:
            print 'import arff failed, currently cannot support ARFF file format'
            return
        DatasetFileBase.__init__(self,filename,extype)
        self.dataname = dataname
        self.comment = comment

    def readlines(self,idx=None):
        """Read from file and split data into examples and labels"""
        fp = open(self.filename,'r')
        (dataname,issparse,alist,data) = arff.arffread(fp)
        fp.close()
        self.dataname = dataname

        #if (alist[0][0]!='label'):
        #    sys.stderr.write('First column of ARFF file needs to be the label\n')
        #    sys.exit(-1)

        if idx is None:
            idx = range(len(data))

        labels = [data[ix][0] for ix in idx]
        labels = array(labels)
        if self.extype == 'vec':
            examples = [data[ix][1:] for ix in idx]
            examples = array(examples).T
            print '%d features, %d examples' % examples.shape
        elif self.extype == 'seq':
            examples = [data[ix][1] for ix in idx]
            print 'sequence length = %d, %d examples' % (len(examples[0]),len(examples))
        elif self.extype == 'mseq':
            examples = [data[ix][1:] for ix in idx]
            printstr = 'sequence lengths = '
            for seq in examples[0]:
                printstr += '%d, ' % len(seq)
            printstr += '%d examples' % len(examples)
            print printstr

        return (examples, labels)

    def writelines(self,examples,labels,idx=None):
        """Merge the examples and labels and write to file"""
        alist = [('label',1,[])]

        if idx is not None:
            examples = examples[idx]
            labels = labels[idx]

        if self.extype == 'vec':
            data = list(concatenate((labels.reshape(len(labels),1),examples.T),axis=1))
            for ix in xrange(examples.shape[0]):
                attname = 'att%d' % ix
                alist.append((attname,1,[]))
        elif self.extype == 'seq':
            data = zip(labels,examples)
            alist.append(('sequence',0,[]))
        elif self.extype == 'mseq':
            data = []
            for ix,curlab in enumerate(labels):
                data.append([curlab]+list(examples[ix]))
            alist.append(('upstream sequence',0,[]))
            alist.append(('downstream sequence',0,[]))

        fp = open(self.filename,'w')
        arff.arffwrite(fp,alist,data,name=self.dataname,comment=self.comment)
        fp.close()


