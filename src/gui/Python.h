#ifdef HAVE_PYTHON

#ifndef __GFPYTHON__
#define __GFPYTHON__

extern "C" {
#include <Python.h>

	PyMODINIT_FUNC initgf(void);
	void exitgf(void);
}
#endif
#endif
