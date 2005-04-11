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

PyObject* CGUIPython::send_command(PyObject* self, PyObject* args)
{
	char *cmd;

	if (!PyArg_ParseTuple(args, "s", &cmd))
		return NULL;

	gui->parse_line(cmd);

	Py_INCREF(Py_None);
	return Py_None;
}

PyObject* CGUIPython::system(PyObject* self, PyObject* args)
{
	char *cmd;

	if (!PyArg_ParseTuple(args, "s", &cmd))
		return NULL;
	::system(cmd);

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
