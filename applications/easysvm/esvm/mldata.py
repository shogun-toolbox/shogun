#!/usr/bin/env python

"""Classes to encapsulate the idea of a dataset in machine learning,
   including file access. Currently this focuses on reading and writing
   transparently to different file formats.

   A dataset is modeled as an (example,label) tuple, each of which is an array.
   The base class doesn't know how to split, so just returns one array.

   The three classes currently implemented use three
   different ways of iterating through files:
   - CSV uses the python module csv's iterator
   - ARFF always reads the whole file, and does a slice
   - FASTA uses a hand crafted while loop that behaves like a generator

   The class DatasetFileARFF is in mldata-arff.py.
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
from numpy import array, concatenate
import csv
import re

try:
    import arff
    have_arff = True
except ImportError:
    have_arff = False


class DatasetFileBase(file):
    """A Base class defining barebones and common behaviour
    """

    def __init__(self,filename,extype):
        """Just the normal file __init__,
        followed by the specific class corresponding to the file extension.

        """
        self.extype = extype
        self.filename = filename


    def readlines(self,idx=None):
        """Read the lines defined by idx (a numpy array).
        Default is read all lines.

        """
        if idx is None:
            data = self.readlines()
        else:
            data = self.readlines()[idx]
            #itertools.islice(open('tempx.txt'), 11, 12).next()
            #file("filename").readlines()[11]
            #linecache.getline(  filename, lineno[, module_globals])
        return data

    def writelines(self,data,idx=None):
        """Write the lines defined by idx (a numpy array).
        Default is write all lines.

        data is assumed to be a numpy array.

        """
        if idx is None:
            self.writelines(data)
        else:
            self.writelines(data[idx])



class DatasetFileCSV(DatasetFileBase):
    """Comma Seperated Values file.

    Labels are in the first column.

    """
    def __init__(self,filename,extype):
        DatasetFileBase.__init__(self,filename,extype)

    def readlines(self,idx=None):
        """Read from file and split data into examples and labels"""
        reader = csv.reader(open(self.filename,'r'), delimiter=',', quoting=csv.QUOTE_NONE)
        labels = []
        examples = []
        for ix,line in enumerate(reader):
            if idx is None or ix in idx:
                labels.append(float(line[0]))
                if self.extype == 'vec':
                    examples.append(array(map(float,line[1:])))
                elif self.extype == 'seq':
                    examples.append(line[1:][0])
                elif self.extype == 'mseq':
                    examples.append(array(line[1:]))

        if self.extype == 'vec':
            examples = array(examples).T
            print '%d features, %d examples' % examples.shape
        elif self.extype == 'seq':
            print 'sequence length = %d, %d examples' % (len(examples[0]),len(examples))
        elif self.extype == 'mseq':
            printstr = 'sequence lengths = '
            for seq in examples[0]:
                printstr += '%d, ' % len(seq)
            printstr += '%d examples' % len(examples)
            print printstr

        return (examples,array(labels))


    def writelines(self,examples,labels,idx=None):
        """Merge the examples and labels and write to file"""
        if idx==None:
            idx = range(len(labels))
        if self.extype == 'seq':
            data = zip(labels[idx],list(array(examples)[idx]))
        if self.extype == 'mseq':
            data = []
            for ix,curlab in enumerate(labels):
                data.append([curlab]+list(examples[ix]))
        elif self.extype == 'vec':
            data = []
            for ix,curlab in enumerate(labels):
                data.append(concatenate((array([curlab]),examples[:,ix].T)))

        fp = open(self.filename,'w')
        writer = csv.writer(fp,delimiter=',',quoting=csv.QUOTE_NONE)
        for ix in idx:
            writer.writerow(data[ix])
        fp.close()




class DatasetFileFASTA(DatasetFileBase):
    """Fasta format file, labels are in the comment after keyword 'label'.
    label=1
    label=-1

    """
    def __init__(self,filename,extype):
        if extype != 'seq':
            print 'Can only write fasta file for sequences!'
            raise IOError
        DatasetFileBase.__init__(self,filename,extype)
        self.fp = None

    def readlines(self,idx=None):
        """Read from file and split data into examples and labels"""
        self.fp = open(self.filename,'r')
        line = self.fp.readline()

        examples = []
        labels = []
        ix = 0
        while True:
            if not line : break
            (ex,lab,line) = self.readline(line)
            if idx is None or ix in idx:
                examples.append(ex)
                labels.append(lab)
            ix += 1

        print 'sequence length = %d, %d examples' % (len(examples[0]),len(examples))
        return (examples,array(labels))

    def writelines(self,examples,labels,idx=None,linelen=60):
        """Write the examples and labels and write to file"""
        if idx==None:
            idx = range(len(labels))

        fp = open(self.filename,'w')
        for ix in idx:
            fp.write('> %d label=%d\n'%(ix,round(labels[ix])))
            for lineidx in xrange(0, len(examples[ix]), linelen):
                fp.write(examples[ix][lineidx:lineidx+linelen] + '\n')
        fp.close()


    def readline(self,line):
        """Reads a fasta entry and returns the label and the sequence"""
        if line[0] == '' : return

        assert(line[0] == '>')
        # Use list comprehension to get the integer that comes after label=
        a = line.split()
        label = float([b.split('=')[1] for b in a if b.split('=')[0]=='label'][0])

        lines = []
        line = self.fp.readline()
        while True:
            if not line : break
            if line[0] == ">": break
            #Remove trailing whitespace, and any internal spaces
            lines.append(line.rstrip().replace(" ",""))
            line = self.fp.readline()

        return (''.join(lines),label,line)


def init_datasetfile(filename,extype):
    """A factory that returns the appropriate class based on the file extension.

    recognised file extensions
    - .csv  : Comma Separated Values
    - .arff : Attribute-Relation File Format (weka)
    - .fa   : Fasta file format (seq only)
    - .fasta: same as above.

    Since the file type does not determine what type of data is actually being used,
    the user has to supply the example type.

    extype can be ('vec','seq','mseq')
    vec - array of floats
    seq - single sequence
    mseq - multiple sequences

    """
    allowedtypes = ('vec','seq','mseq')
    assert(extype in allowedtypes)
    # map the file extensions to the relevant classes
    _format2dataset = {'csv'   : DatasetFileCSV,
                       'fa'    : DatasetFileFASTA,
                       'fasta' : DatasetFileFASTA,
                       }
    if have_arff:
        from esvm.mldata_arff import DatasetFileARFF
        _format2dataset['arff'] = DatasetFileARFF

    extension = detect_extension(filename)
    return _format2dataset[extension](filename,extype)


def detect_extension(filename):
    """Get the file extension"""
    if filename.count('.') > 1:
        print 'WARNING: %s has more than one . using last one' % filename
    detect_ext = filename.split('.')[-1]
    if have_arff:
        known_ext = ('csv','arff','fasta','fa')
    else:
        known_ext = ('csv','fasta','fa')

    if detect_ext not in known_ext:
        print 'WARNING: %s is an unknown file extension, defaulting to csv' % detect_ext
        detect_ext = 'csv'

    if detect_ext == 'csv':
        fasta_flag = 0
        arff_flag = 0
        run_c = 0
        f = open(filename,'r')
        for line in f:
           line = line.strip()
           if re.match(r'^>',line):
               fasta_flag = 1
               break
           if re.match(r'^@',line):
               arff_flag = 1
               break
           if run_c == 5:
               break
        f.close()
        if fasta_flag == 1:
           detect_ext = 'fasta'
        elif arff_flag == 1:
           detect_ext = 'arff'
        else:
           detect_ext = 'csv'

    return detect_ext


def convert(infile,outfile,extype):
    """Copy data from infile to outfile, possibly converting the file format."""
    fp1 = init_datasetfile(infile,extype)
    (examples,labels) = fp1.readlines()
    fp2 = init_datasetfile(outfile,extype)
    fp2.writelines(examples,labels)

