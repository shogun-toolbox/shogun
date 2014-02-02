#!/usr/bin/env python

from modshogun import WeightedDegreeStringKernel, LinearKernel, PolyKernel, GaussianKernel, CTaxonomy
from modshogun import CombinedKernel, WeightedDegreeRBFKernel
from modshogun import StringCharFeatures, RealFeatures, CombinedFeatures, StringWordFeatures, SortWordString
from modshogun import DNA, PROTEIN, Labels
from modshogun import WeightedDegreeStringKernel, CombinedKernel, WeightedCommWordStringKernel, WeightedDegreePositionStringKernel
from modshogun import StringCharFeatures, DNA, StringWordFeatures, CombinedFeatures

from modshogun import MSG_DEBUG
from modshogun import RealFeatures, BinaryLabels, DNA, Alphabet
from modshogun import WeightedDegreeStringKernel, GaussianKernel
from modshogun import SVMLight
from numpy import concatenate, ones
from numpy.random import randn, seed
import numpy
import sys
import types
import random
import bz2
import pickle
import inspect

###################################################
#             Random Data
###################################################

def generate_random_string(length, number):
    """
    generate sample over alphabet
    """

    dat = []

    alphabet = "AGTC"

    for i in range(number):
        dat.append("".join([random.choice(alphabet) for j in range(length)]))

    return dat


def generate_random_data(number):
    """
    create random examples and labels
    """

    labels = numpy.array([random.choice([-1.0, 1.0]) for i in range(number)])
    examples = numpy.array(generate_random_string(22, number))

    return examples, labels


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
    except IOError as details:
        sys.stderr.write('File ' + filename + ' cannot be written\n')
        sys.stderr.write(details)
        return

    pickle.dump(myobj, f, protocol=2)
    f.close()


def load(filename):
    """
    Load from filename using pickle

    @param filename: name of file to load from
    @type filename: str
    """

    try:
        f = bz2.BZ2File(filename, 'rb')
    except IOError as details:
        sys.stderr.write('File ' + filename + ' cannot be read\n')
        sys.stderr.write(details)
        return

    myobj = pickle.load(f)
    f.close()
    return myobj


def get_spectrum_features(data, order=3, gap=0, reverse=True):
    """
    create feature object used by spectrum kernel
    """

    charfeat = StringCharFeatures(data, DNA)
    feat = StringWordFeatures(charfeat.get_alphabet())
    feat.obtain_from_char(charfeat, order-1, order, gap, reverse)
    preproc = SortWordString()
    preproc.init(feat)
    feat.add_preprocessor(preproc)
    feat.apply_preprocessor()

    return feat


def get_wd_features(data, feat_type="dna"):
    """
    create feature object for wdk
    """
    if feat_type == "dna":
        feat = StringCharFeatures(DNA)
    elif feat_type == "protein":
        feat = StringCharFeatures(PROTEIN)
    else:
        raise Exception("unknown feature type")
    feat.set_features(data)

    return feat


def construct_features(features):
    """
    makes a list
    """

    feat_all = [inst for inst in features]
    feat_lhs = [inst[0:15] for inst in features]
    feat_rhs = [inst[15:] for inst in features]

    feat_wd = get_wd_features(feat_all)
    feat_spec_1 = get_spectrum_features(feat_lhs, order=3)
    feat_spec_2 = get_spectrum_features(feat_rhs, order=3)

    feat_comb = CombinedFeatures()
    feat_comb.append_feature_obj(feat_wd)
    feat_comb.append_feature_obj(feat_spec_1)
    feat_comb.append_feature_obj(feat_spec_2)

    return feat_comb

parameter_list = [[200, 1, 100]]

def serialization_string_kernels_modular(n_data, num_shifts, size):
    """
    serialize svm with string kernels
    """

    ##################################################
    # set up toy data and svm
    train_xt, train_lt = generate_random_data(n_data)
    test_xt, test_lt = generate_random_data(n_data)

    feats_train = construct_features(train_xt)
    feats_test = construct_features(test_xt)

    max_len = len(train_xt[0])
    kernel_wdk = WeightedDegreePositionStringKernel(size, 5)
    shifts_vector = numpy.ones(max_len, dtype=numpy.int32)*num_shifts
    kernel_wdk.set_shifts(shifts_vector)

    ########
    # set up spectrum
    use_sign = False
    kernel_spec_1 = WeightedCommWordStringKernel(size, use_sign)
    kernel_spec_2 = WeightedCommWordStringKernel(size, use_sign)

    ########
    # combined kernel
    kernel = CombinedKernel()
    kernel.append_kernel(kernel_wdk)
    kernel.append_kernel(kernel_spec_1)
    kernel.append_kernel(kernel_spec_2)

    # init kernel
    labels = BinaryLabels(train_lt);

    svm = SVMLight(1.0, kernel, labels)
    #svm.io.set_loglevel(MSG_DEBUG)
    svm.train(feats_train)

    ##################################################
    # serialize to file

    fn = "serialized_svm.bz2"
    #print("serializing SVM to file", fn)
    save(fn, svm)

    ##################################################
    # unserialize and sanity check

    #print("unserializing SVM")
    svm2 = load(fn)


    #print("comparing predictions")
    out =  svm.apply(feats_test).get_labels()
    out2 =  svm2.apply(feats_test).get_labels()

    # assert outputs are close
    for i in range(len(out)):
        assert abs(out[i] - out2[i] < 0.000001)

    #print("all checks passed.")

    return out,out2


if __name__=='__main__':
    serialization_string_kernels_modular(*parameter_list[0])

