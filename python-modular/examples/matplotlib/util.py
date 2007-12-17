""" Utilities for matplotlib examples """

import sys
from pylab import get_current_fig_manager

QUITKEY='q'

def quit (event):
	if event.key==QUITKEY or event.key==QUITKEY.upper():
		sys.exit(0)

def set_title (title):
	quitmsg=" (press '"+QUITKEY+"' to quit)"
	complete=title+quitmsg
	manager=get_current_fig_manager()

	# now we have to wrap the toolkit
	if hasattr(manager, 'window'):
		if hasattr(manager.window, 'setCaption'): # QT
			manager.window.setCaption(complete)
		if hasattr(manager.window, 'set_title'): # GTK
			manager.window.set_title(complete)
		elif hasattr(manager.window, 'title'): # TK
			manager.window.title(complete)

