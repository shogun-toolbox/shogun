from shogun.Features import *
from shogun.Library import MSG_DEBUG
from shogun.Features import StringCharFeatures, Labels, DNA, Alphabet
from shogun.Kernel import WeightedDegreeStringKernel, GaussianKernel
from shogun.Classifier import SVMLight
from numpy import *
from numpy.random import randn

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


##################################################
num=10
dist=1
width=2.1

traindata_real=concatenate((randn(2,num)-dist, randn(2,num)+dist), axis=1)
testdata_real=concatenate((randn(2,num)-dist, randn(2,num)+dist), axis=1);

trainlab=concatenate((-ones(num), ones(num)));
testlab=concatenate((-ones(num), ones(num)));

feats_train=RealFeatures(traindata_real);
feats_test=RealFeatures(testdata_real);
kernel=GaussianKernel(feats_train, feats_train, width);
kernel.io.set_loglevel(MSG_DEBUG)

labels=Labels(trainlab);

svm=SVMLight(2, kernel, labels)
svm.train()
svm.io.set_loglevel(MSG_DEBUG)

##################################################

print "labels:"
print labels.to_string()

print "features"
print feats_train.to_string()

print "kernel"
print kernel.to_string()

print "svm"
print svm.to_string()

print "#################################"

fn = "serialized_svm.bz2"
print "serializing SVM to file", fn

save(fn, svm)

print "#################################"

print "unserializing SVM"
svm2 = load(fn)


print "#################################"
print "comparing training"

svm2.train()

print "objective before serialization:", svm.get_objective()
print "objective after serialization:", svm2.get_objective()

