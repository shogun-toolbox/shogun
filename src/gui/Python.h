#ifdef HAVE_PYTHON

#ifndef __GFPYTHON__
#define __GFPYTHON__

extern "C" {
#include <Python.h>

	PyMODINIT_FUNC initsg(void);
	void exitsg(void);
}
#endif
#endif
