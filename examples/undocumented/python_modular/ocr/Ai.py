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

# Since python 3.0 we need to import _THREAD, earlier versions are
# needing THREAD.  If THREAD is not supported by OS we are using the
# fallback solution DUMMY_THREAD.
try:
    import _thread as thr
except ImportError:
    try:
        import thread as thr
    except ImportError:
        import dummy_thread as thr

import numpy as np
import gobject as go
import gzip as gz
import tempfile as tmp
import pickle as pkl

import common as com

class Ai:
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

    def _load_classifier(self, x_fname, y_fname, main_window):
        go.idle_add(main_window.idle_show_wait)
        com.dispatch()
        try:
            self.load_train_data(x_fname, y_fname)
            self.read_svm()
        except:
            go.idle_add(main_window.idle_enable_go, True)
            raise
        go.idle_add(main_window.set_title,
                    "%s - %s" % (
                main_window.TITLE, self.get_config_str()
                ))
        go.idle_add(main_window.idle_enable_go, False)
        thr.exit()

    def load_classifier(self, x_fname, y_fname, main_window):
        thr.start_new_thread(self._load_classifier,
                             (x_fname, y_fname, main_window))

    def classify(self, matrix):
        cl = self.svm.classify(
            RealFeatures(
                np.reshape(matrix, newshape=(com.FEATURE_DIM, 1),
                           order='F')
                )
            ).get_label(0)

        return int(cl + 1.0) % 10

# Error stuff begin
# ********************************************************************

    def _get_error(self, x, y):
        self.svm.io.enable_progress()
        l = self.svm.classify(RealFeatures(x)).get_labels()
        self.svm.io.disable_progress()

        return 1.0 - np.mean(l == y)

    def _show_error(self, main_window, x, y, str):
        go.idle_add(main_window.idle_show_wait)
        com.dispatch()
        try:
            e = self._get_error(x, y)
        except:
            go.idle_add(main_window.idle_enable_go, True)
            raise
        go.idle_add(main_window.idle_enable_go, False)
        go.idle_add(main_window.idle_info_dialog,
                    "The %s error is: %.2f%%" % (str, e*100.0))
        thr.exit()

    def get_train_error(self):
        return self._get_error(self.x_test, self.y_test)

    def show_train_error(self, main_window):
        thr.start_new_thread(self._show_error, (main_window,
                                                self.x_test,
                                                self.y_test,
                                                "training"))

    def _noised_x(self, noise):
        return np.array((np.random.rand(*self.x_test.shape) < noise) \
                            .__xor__(self.x_test > com.NEAR_ZERO_POS),
                        dtype=np.float)

    def get_test_error(self, noise):
        return self._get_error(self._noised_x(noise), self.y_test)

    def show_test_error(self, main_window, noise):
        thr.start_new_thread(self._show_error, (main_window,
                                                self._noised_x(noise),
                                                self.y_test,
                                                "test"))

# Error stuff end
# ********************************************************************

    def get_config_str(self):
        return "C: %.2f, epsilon: %.2e, kernel-width: %.2f" \
            % (self.svm.get_C1(), self.svm.get_epsilon(),
               self.kernel.get_width())
