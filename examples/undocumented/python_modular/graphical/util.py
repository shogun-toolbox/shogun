""" Utilities for matplotlib examples """

import pylab
from numpy import ones, array, double, meshgrid, reshape, linspace, \
	concatenate, ravel, pi, sinc
from numpy.random import randn, rand
from modshogun import BinaryLabels, RegressionLabels, RealFeatures, SparseRealFeatures

QUITKEY='q'
NUM_EXAMPLES=100
DISTANCE=2

def quit (event):
	if event.key==QUITKEY or event.key==QUITKEY.upper():
		pylab.close()

def set_title (title):
	quitmsg=" (press '"+QUITKEY+"' to quit)"
	complete=title+quitmsg
	manager=pylab.get_current_fig_manager()

	# now we have to wrap the toolkit
	if hasattr(manager, 'window'):
		if hasattr(manager.window, 'setCaption'): # QT
			manager.window.setCaption(complete)
		if hasattr(manager.window, 'set_title'): # GTK
			manager.window.set_title(complete)
		elif hasattr(manager.window, 'title'): # TK
			manager.window.title(complete)


def get_realdata(positive=True):
	if positive:
		return randn(2, NUM_EXAMPLES)+DISTANCE
	else:
		return randn(2, NUM_EXAMPLES)-DISTANCE


def get_realfeatures(pos, neg):
	arr=array((pos, neg))
	features = concatenate(arr, axis=1)
	return RealFeatures(features)


def get_labels(raw=False, type='binary'):
	data = concatenate(array(
		(-ones(NUM_EXAMPLES, dtype=double), ones(NUM_EXAMPLES, dtype=double))
	))
	if raw:
		return data
	else:
		if type == 'binary':
			return BinaryLabels(data)
		if type == 'regression':
			return RegressionLabels(data)
		return None


def compute_output_plot_isolines(classifier, kernel=None, train=None, sparse=False, pos=None, neg=None, regression=False):
	size=100
	if pos is not None and neg is not None:
		x1_max=max(1.2*pos[0,:])
		x1_min=min(1.2*neg[0,:])
		x2_min=min(1.2*neg[1,:])
		x2_max=max(1.2*pos[1,:])
		x1=linspace(x1_min, x1_max, size)
		x2=linspace(x2_min, x2_max, size)
	else:
		x1=linspace(-5, 5, size)
		x2=linspace(-5, 5, size)

	x, y=meshgrid(x1, x2)

	dense=RealFeatures(array((ravel(x), ravel(y))))
	if sparse:
		test=SparseRealFeatures()
		test.obtain_from_simple(dense)
	else:
		test=dense

	if kernel and train:
		kernel.init(train, test)
	else:
		classifier.set_features(test)

	labels = None
	if regression:
		labels=classifier.apply().get_labels()
	else:
		labels=classifier.apply().get_values()
	z=labels.reshape((size, size))

	return x, y, z


def get_sinedata():
	x=4*rand(1, NUM_EXAMPLES)-DISTANCE
	x.sort()
	y=sinc(pi*x)+0.1*randn(1, NUM_EXAMPLES)

	return x, y


def compute_output_plot_isolines_sine(classifier, kernel, train, regression=False):
	x=4*rand(1, 500)-2
	x.sort()
	test=RealFeatures(x)
	kernel.init(train, test)

	if regression:
		y=classifier.apply().get_labels()
	else:
		y=classifier.apply().get_values()

	return x, y
