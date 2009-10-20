from shogun.Features import *
from shogun.Library import MSG_DEBUG
from shogun.Features import StringCharFeatures, Labels, DNA, Alphabet
from shogun.Kernel import WeightedDegreeStringKernel
from shogun.Classifier import SVMLight
import numpy

import sys
import types
import random
import bz2
import cPickle
import inspect



def save(filename, myobj):
    """
    save object to file using pickle
    
    @param filename: name of destination file
    @type filename: str
    @param myobj: object to save (has to be pickleable)
    @type myobj: obj
    """

    try:
        f = bz2.BZ2File(filename, 'wb')
    except IOError, details:
        sys.stderr.write('File ' + filename + ' cannot be written\n')
        sys.stderr.write(details)
        return

    cPickle.dump(myobj, f, protocol=2)
    f.close()



def load(filename):
    """
    Load from filename using pickle
    
    @param filename: name of file to load from
    @type filename: str
    """
    
    try:
        f = bz2.BZ2File(filename, 'rb')
    except IOError, details:
        sys.stderr.write('File ' + filename + ' cannot be read\n')
        sys.stderr.write(details)
        return

    myobj = cPickle.load(f)
    f.close()
    return myobj



degree=3
fm_train_dna=['CGCACGTACGTAGCTCGAT',
		      'CGACGTAGTCGTAGTCGTA',
		      'CGACGGGGGGGGGGTCGTA',
		      'CGACCTAGTCGTAGTCGTA',
		      'CGACCACAGTTATATAGTA',
		      'CGACGTAGTCGTAGTCGTA',
		      'CGACGTAGTTTTTTTCGTA',
		      'CGACGTAGTCGTAGCCCCA',
		      'CAAAAAAAAAAAAAAAATA',
		      'CGACGGGGGGGGGGGCGTA']
label_train_dna=numpy.array(5*[-1.0] + 5*[1.0])
fm_test_dna=['AGCACGTACGTAGCTCGAT',
		      'AGACGTAGTCGTAGTCGTA',
		      'CAACGGGGGGGGGGTCGTA',
		      'CGACCTAGTCGTAGTCGTA',
		      'CGAACACAGTTATATAGTA',
		      'CGACCTAGTCGTAGTCGTA',
		      'CGACGTGGGGTTTTTCGTA',
		      'CGACGTAGTCCCAGCCCCA',
		      'CAAAAAAAAAAAACCAATA',
		      'CGACGGCCGGGGGGGCGTA']
label_test_dna=numpy.array(5*[-1.0] + 5*[1.0])


##################################################

alpha = Alphabet(DNA)
alpha.io.set_loglevel(MSG_DEBUG)

feats_train=StringCharFeatures(DNA)
feats_train.set_features(fm_train_dna)
feats_test=StringCharFeatures(DNA)
feats_test.set_features(fm_test_dna)

kernel=WeightedDegreeStringKernel(feats_train, feats_train, degree)
kernel.io.set_loglevel(MSG_DEBUG)

C=10
epsilon=1e-5
num_threads=1
labels=Labels(label_train_dna)

svm=SVMLight(C, kernel, labels)
svm.set_qpsize(3)
svm.set_linear_term(-numpy.array([1,2,3,4,5,6,7,8,7,6], dtype=numpy.double));
svm.set_epsilon(epsilon)
svm.parallel.set_num_threads(num_threads)
svm.train()
svm.io.set_loglevel(MSG_DEBUG)

##################################################

print "alphabet:"
print alpha.to_string()

print "labels:"
print labels.to_string()

print "features"
print feats_train.to_string()

print "kernel"
print kernel.to_string()

print "svm"
print svm.to_string()

fn = "/tmp/myawesomesvm.bz2"
save(fn, svm)

svm2 = load(fn)

svm2.train()


