""" Utilities for matplotlib examples """

import pylab
import numpy as np
from numpy.random import randn, rand
import shogun as sg

QUITKEY = 'q'
NUM_EXAMPLES = 100
DISTANCE = 2


def quit(event):
    if event.key == QUITKEY or event.key == QUITKEY.upper():
        pylab.close()


def set_title(title):
    quitmsg = " (press '" + QUITKEY + "' to quit)"
    complete = title + quitmsg
    manager = pylab.get_current_fig_manager()

    # now we have to wrap the toolkit
    if hasattr(manager, 'window'):
        if hasattr(manager.window, 'setCaption'):  # QT
            manager.window.setCaption(complete)
        if hasattr(manager.window, 'set_title'):  # GTK
            manager.window.set_title(complete)
        elif hasattr(manager.window, 'title'):  # TK
            manager.window.title(complete)


def get_realdata(positive=True):
    if positive:
        return randn(2, NUM_EXAMPLES) + DISTANCE
    else:
        return randn(2, NUM_EXAMPLES) - DISTANCE


def get_realfeatures(pos, neg):
    arr = np.array((pos, neg))
    features = np.concatenate(arr, axis=1)
    return sg.create_features(features)


def get_labels(raw=False):
    data = np.concatenate(np.array(
        (-np.ones(NUM_EXAMPLES, dtype=np.double), np.ones(NUM_EXAMPLES, dtype=np.double))
    ))
    if raw:
        return data
    else:
        return sg.create_labels(data)


def compute_output_plot_isolines(classifier, kernel=None, train=None, sparse=False, pos=None, neg=None,
                                 regression=False):
    size = 100
    if pos is not None and neg is not None:
        x1_max = max(1.2 * pos[0, :])
        x1_min = min(1.2 * neg[0, :])
        x2_min = min(1.2 * neg[1, :])
        x2_max = max(1.2 * pos[1, :])
        x1 = np.linspace(x1_min, x1_max, size)
        x2 = np.linspace(x2_min, x2_max, size)
    else:
        x1 = np.linspace(-5, 5, size)
        x2 = np.linspace(-5, 5, size)

    x, y = np.meshgrid(x1, x2)

    dense = sg.create_features(np.array((np.ravel(x), np.ravel(y))))
    if sparse:
        test = sg.SparseRealFeatures()
        test.obtain_from_simple(dense)
    else:
        test = dense

    if kernel and train:
        kernel.init(train, test)
    else:
        classifier.put('features', test)

    labels = None
    if regression:
        labels = classifier.apply().get('labels')
    else:
        labels = classifier.apply().get('current_values')
    z = labels.reshape((size, size))

    return x, y, z


def get_sinedata():
    x = 4 * rand(1, NUM_EXAMPLES) - DISTANCE
    x.sort()
    y = np.sinc(np.pi * x) + 0.1 * randn(1, NUM_EXAMPLES)

    return x, y


def compute_output_plot_isolines_sine(classifier, kernel, train, regression=False):
    x = 4 * rand(1, 500) - 2
    x.sort()
    test = sg.create_features(x)
    kernel.init(train, test)

    if regression:
        y = classifier.apply().get('labels')
    else:
        y = classifier.apply().get('values')

    return x, y
