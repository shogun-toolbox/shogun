# File   : $HeadURL$
# Version: $Id$

from modshogun import RealFeatures, MulticlassLabels
from modshogun import GaussianKernel
from modshogun import GMNPSVM

import numpy as np
import gzip as gz
import pickle as pkl

import common as com

class Ai:
    def __init__(self):
        self.x = None
        self.y = None

        self.x_test = None
        self.y_test = None

        self.svm = None

    def load_train_data(self, x_fname, y_fname):
        Ai.__init__(self)

        self.x = np.loadtxt(x_fname)
        self.y = np.loadtxt(y_fname) - 1.0

        self.x_test = self.x
        self.y_test = self.y

    def _svm_new(self, kernel_width, c, epsilon):
        if self.x == None or self.y == None:
            raise Exception("No training data loaded.")

        x = RealFeatures(self.x)
        y = MulticlassLabels(self.y)

        self.svm = GMNPSVM(c, GaussianKernel(x, x, kernel_width), y)
        self.svm.set_epsilon(epsilon)

    def write_svm(self):
        gz_stream = gz.open(com.TRAIN_SVM_FNAME_GZ, 'wb', 9)
        pkl.dump(self.svm, gz_stream)
        gz_stream.close()

    def read_svm(self):
        gz_stream = gz.open(com.TRAIN_SVM_FNAME_GZ, 'rb')
        self.svm = pkl.load(gz_stream)
        gz_stream.close()

    def enable_validation(self, train_frac):
        x = self.x
        y = self.y

        idx = np.arange(len(y))
        np.random.shuffle(idx)
        train_idx=idx[:np.floor(train_frac*len(y))]
        test_idx=idx[np.ceil(train_frac*len(y)):]

        self.x = x[:,train_idx]
        self.y = y[train_idx]
        self.x_test = x[:,test_idx]
        self.y_test = y[test_idx]

    def train(self, kernel_width, c, epsilon):
        self._svm_new(kernel_width, c, epsilon)

        x = RealFeatures(self.x)
        self.svm.io.enable_progress()
        self.svm.train(x)
        self.svm.io.disable_progress()

    def load_classifier(self): self.read_svm()

    def classify(self, matrix):
        cl = self.svm.apply(
            RealFeatures(
                np.reshape(matrix, newshape=(com.FEATURE_DIM, 1),
                           order='F')
                )
            ).get_label(0)

        return int(cl + 1.0) % 10

    def get_test_error(self):
        self.svm.io.enable_progress()
        l = self.svm.apply(RealFeatures(self.x_test)).get_labels()
        self.svm.io.disable_progress()

        return 1.0 - np.mean(l == self.y_test)
