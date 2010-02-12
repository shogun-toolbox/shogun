""" Utilities for matplotlib examples """

import pylab
from numpy import ones, array, meshgrid, linspace, concatenate, ravel, min, max
from numpy.random import randn

QUITKEY='q'
NUM_EXAMPLES=200
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


def get_traindata():
	return concatenate(
		(randn(2, NUM_EXAMPLES)+DISTANCE, randn(2, NUM_EXAMPLES)-DISTANCE),
		axis=1)


def get_meshgrid(traindata):
	x1=linspace(1.2*min(traindata), 1.2*max(traindata), 50)
	x2=linspace(1.2*min(traindata), 1.2*max(traindata), 50)
	return meshgrid(x1,x2)


def get_testdata(x, y):
	return array((ravel(x), ravel(y)))


def get_labels(raw=False):
	return concatenate(
		(-ones([1, NUM_EXAMPLES]), ones([1, NUM_EXAMPLES])),
		axis=1)[0]

