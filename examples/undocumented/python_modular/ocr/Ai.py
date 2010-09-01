# File   : $HeadURL$
# Version: $Id$

# For reasons of development ...
try:
    from Features import RealFeatures, Labels
    from Kernel import GaussianKernel
    from Classifier import GMNPSVM
except ImportError:
    from shogun.Features import RealFeatures, Labels
    from shogun.Kernel import GaussianKernel
    from shogun.Classifier import GMNPSVM

import numpy as np
import gzip as gz
import tempfile as tmp
import pickle as pkl

import common as com

class Ai:
    KERNEL_SIZE = 10

    def __init__(self):
        self.x = None
        self.y = None

        self.x_test = None
        self.y_test = None

        self.svm = None
        self.kernel = None

        self.kernel_width = None
        self.c = None
        self.epsilon = None

    def load_train_data(self, x_fname, y_fname):
        Ai.__init__(self)

        self.x = np.loadtxt(x_fname)
        self.y = np.loadtxt(y_fname) - 1.0

        self.x_test = self.x
        self.y_test = self.y

    def _svm_new(self, kernel_width, c, epsilon):
        if self.x == None or self.y == None:
            raise Exception("No training data loaded.")

        self.kernel_width = kernel_width
        self.c = c
        self.epsilon = epsilon

        x = RealFeatures(self.x)
        y = Labels(self.y)

        self.kernel = GaussianKernel(x, x, self.kernel_width)

        self.svm = GMNPSVM(self.c, self.kernel, y)
        self.svm.set_epsilon(self.epsilon)

    def write_svm(self):
        fstream = open(com.TRAIN_PARAMS_FNAME, 'wb')
        pkl.dump({'kernel_width': self.kernel_width,
                  'c': self.c,
                  'epsilon': self.epsilon},
                 fstream)
        fstream.close()

        asc_stream = tmp.TemporaryFile('w+b')
        gz_stream = gz.open(com.TRAIN_SVM_FNAME_GZ, 'wb', 9)

        self.svm.save(asc_stream)
        asc_stream.seek(0)
        gz_stream.write(asc_stream.read())

        gz_stream.close()
        asc_stream.close()

    def read_svm(self):
        fstream = open(com.TRAIN_PARAMS_FNAME, 'rb')
        params = pkl.load(fstream)
        fstream.close()

        self._svm_new(kernel_width=params['kernel_width'],
                      c=params['c'],
                      epsilon=params['epsilon'])

        gz_stream = gz.open(com.TRAIN_SVM_FNAME_GZ, 'rb')
        asc_stream = tmp.TemporaryFile('w+b')

        asc_stream.write(gz_stream.read())
        asc_stream.seek(0)
        self.svm.load(asc_stream)

        asc_stream.close()
        gz_stream.close()

    def enable_validation(self, train_frac):
        x = self.x
        y = self.y
        self.x = np.empty((com.FEATURE_DIM, 0))
        self.y = np.empty((0, ))
        self.x_test = np.empty((com.FEATURE_DIM, 0))
        self.y_test = np.empty((0, ))

        for i in range(x.shape[1]):
            if np.random.rand() < train_frac:
                self.x = np.append(self.x, np.transpose([x[:, i]]),
                                   axis=1)
                self.y = np.append(self.y, [y[i]], axis=0)
            else:
                self.x_test = np.append(self.x_test,
                                        np.transpose([x[:, i]]),
                                        axis=1)
                self.y_test = np.append(self.y_test, [y[i]], axis=0)

    def train(self, kernel_width, c, epsilon):
        self._svm_new(kernel_width, c, epsilon)

        x = RealFeatures(self.x)
        self.svm.io.enable_progress()
        self.svm.train(x)
        self.svm.io.disable_progress()

    def load_classifier(self, x_fname, y_fname, main_window):
        self.load_train_data(x_fname, y_fname)
        self.read_svm()
        main_window.set_title("%s - %s" % (
                main_window.TITLE, self.get_config_str())
                              )

    def classify(self, matrix):
        cl = self.svm.classify(
            RealFeatures(
                np.reshape(matrix, newshape=(com.FEATURE_DIM, 1),
                           order='F')
                )
            ).get_label(0)

        return int(cl + 1.0) % 10

    def get_train_error(self):
        self.svm.io.enable_progress()
        l = self.svm.classify(RealFeatures(self.x_test)).get_labels()
        self.svm.io.disable_progress()

        return 1.0 - np.mean(l == self.y_test)

    def get_config_str(self):
        return "C: %.2f, epsilon: %.2e, kernel-width: %.2f" \
            % (self.svm.get_C1(), self.svm.get_epsilon(),
               self.kernel.get_width())
