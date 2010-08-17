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

import common as com

class Ai:
    KERNEL_WIDTH = 16.0
    C = 1.0
    EPSILON = 1e-8

    def __init__(self):
        self.x = None
        self.y = None
        self.svm = None

    def _load_train_data(self, x_fname, y_fname):
        Ai.__init__(self)

        self.x = np.loadtxt(x_fname)
        self.y = np.loadtxt(y_fname) - 1.0

        x = RealFeatures(self.x)
        y = Labels(self.y)

        kernel = GaussianKernel(x, x, self.KERNEL_WIDTH)

        self.svm = GMNPSVM(self.C, kernel, y)
        self.svm.set_epsilon(self.EPSILON)

    def _load_train_data_train(self, x_fname, y_fname, main_window):
        go.idle_add(main_window.idle_show_wait)
        com.dispatch()
        try:
            self._load_train_data(x_fname, y_fname)

            x = RealFeatures(self.x)
            self.svm.io.enable_progress()
            self.svm.train(x)
            self.svm.io.disable_progress()

            if com.TRAIN_WRITE_GZ:
                asc_stream = tmp.TemporaryFile('w+b')
                gz_stream = gz.open(com.TRAIN_SVM_FNAME_GZ, 'wb', 9)

                self.svm.save(asc_stream)
                asc_stream.seek(0)
                gz_stream.write(asc_stream.read())

                gz_stream.close()
                asc_stream.close()
        except:
            go.idle_add(main_window.idle_enable_go, True)
            raise
        go.idle_add(main_window.idle_enable_go, False)
        thr.exit()

    def load_train_data_train(self, x_fname, y_fname, main_window):
        thr.start_new_thread(self._load_train_data_train,
                             (x_fname, y_fname, main_window))

    def _load_classifier(self, x_fname, y_fname, main_window):
        go.idle_add(main_window.idle_show_wait)
        com.dispatch()
        try:
            self._load_train_data(x_fname, y_fname)

            gz_stream = gz.open(com.TRAIN_SVM_FNAME_GZ, 'rb')
            asc_stream = tmp.TemporaryFile('w+b')

            asc_stream.write(gz_stream.read())
            asc_stream.seek(0)
            self.svm.load(asc_stream)

            asc_stream.close()
            gz_stream.close()
        except:
            go.idle_add(main_window.idle_enable_go, True)
            raise
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

    def _show_error(self, main_window, x, y, str):
        go.idle_add(main_window.idle_show_wait)
        com.dispatch()
        try:
            self.svm.io.enable_progress()
            l = self.svm.classify(RealFeatures(x)) \
                .get_labels()
            self.svm.io.disable_progress()
        except:
            go.idle_add(main_window.idle_enable_go, True)
            raise
        go.idle_add(main_window.idle_enable_go, False)
        go.idle_add(main_window.idle_info_dialog,
                    "The %s error is: %.2f%%"
                    % (str,
                       (100.0*(1.0 - np.mean(l == y)))
                       )
                    )
        thr.exit()

    def show_train_error(self, main_window):
        thr.start_new_thread(self._show_error, (main_window,
                                                self.x, self.y,
                                                "training"))

    def show_test_error(self, main_window, noise):
        x = np.array((np.random.rand(*self.x.shape) < noise) \
                         .__xor__(self.x > com.NEAR_ZERO_POS),
                     dtype=np.float)
        thr.start_new_thread(self._show_error, (main_window,
                                                x, self.y,
                                                "test"))
