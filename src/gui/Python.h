#ifdef HAVE_PYTHON

#ifndef __GFPYTHON__
#define __GFPYTHON__

#include <Python.h>

extern "C" {
	PyMODINIT_FUNC initgf(void);
	void exitgf(void);
}
#endif
#endif
