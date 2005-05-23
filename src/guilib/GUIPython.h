#include "lib/config.h"

#ifdef HAVE_PYTHON
#ifndef __GUIPYTHON_H_
#define __GUIPYTHON_H_

#include <Python.h>

class CGUIPython
{
public:
	CGUIPython();
	~CGUIPython();

	// this simply sends a cmd to genefinder
	// 		gf('send_command', 'cmdline');
	static PyObject* py_send_command(PyObject* self, PyObject* args);
	static PyObject* py_system(PyObject* self, PyObject* args);

	static PyObject* py_get_kernel_matrix(PyObject* self, PyObject* args);
	static PyObject* py_set_kernel_matrix(PyObject* self, PyObject* args);

	static PyObject* py_set_features(PyObject* self, PyObject* args);

	static PyObject* py_test(PyObject* self, PyObject* args);
};
#endif
#endif
