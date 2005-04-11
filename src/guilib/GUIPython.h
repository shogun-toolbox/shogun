#include "lib/config.h"

#ifdef HAVE_PYTHON
#ifndef __GUIPYTHON_H_
#define __GUIPYTHON_H_
#include <Python.h>
#include "lib/common.h"

class CGUIPython
{
public:
	CGUIPython();
	~CGUIPython();

	// this simply sends a cmd to genefinder
	// 		gf('send_command', 'cmdline');
	static PyObject* send_command(PyObject* self, PyObject* args);
	static PyObject* system(PyObject* self, PyObject* args);
};
#endif
#endif
