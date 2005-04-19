#include "lib/config.h"

#ifdef HAVE_PYTHON
#include <Python.h>
#include <numarray/arrayobject.h>
#include <stdlib.h>

#include "gui/TextGUI.h"
#include "guilib/GUIPython.h"

extern CTextGUI* gui;

CGUIPython::CGUIPython()
{
}

CGUIPython::~CGUIPython()
{
}

PyObject* CGUIPython::py_send_command(PyObject* self, PyObject* args)
{
	char *cmd;

	if (!PyArg_ParseTuple(args, "s", &cmd))
		return NULL;

	gui->parse_line(cmd);

	Py_INCREF(Py_None);
	return Py_None;
}

PyObject* CGUIPython::py_system(PyObject* self, PyObject* args)
{
	char *cmd;

	if (!PyArg_ParseTuple(args, "s", &cmd))
		return NULL;
	::system(cmd);

	Py_INCREF(Py_None);
	return Py_None;
}

PyObject* CGUIPython::py_get_kernel_matrix(PyObject* self, PyObject* args)
{
	PyArrayObject* convolved;
	if (convolved == Py_None)
		convolved = (PyArrayObject *) PyArray_FromDims(
				data->nd, data->dimensions, tFloat64);
	else
		convolved = (PyArrayObject *) PyArray_ContiguousFromObject(oconvolved, tFloat64, 2, 2);
	Py_INCREF(Py_None);
	return Py_None;
}

//static PyObject * gfpython_system(PyObject *self, PyObject *args)
//{
//	char *command;
//	int sts;
//
//	if (!PyArg_ParseTuple(args, "s", &command))
//	{
//		PyErr_SetString(PyExc_RuntimeError, "du bist doch doof");
//		return NULL;
//	}
//	sts = system(command);
//	return Py_BuildValue("i", sts);
//}

#endif
