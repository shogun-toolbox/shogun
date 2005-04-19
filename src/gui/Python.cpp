#include "lib/config.h"

#ifdef HAVE_PYTHON

#include <Python.h>
#include <numarray/libnumarray.h>
#include <numarray/arrayobject.h>

#include "guilib/GUIPython.h"
#include "gui/Python.h"
#include "gui/TextGUI.h"

static CGUIPython gfpy;
extern CTextGUI* gui;

static PyMethodDef gfpythonmethods[] = {
    {"send_command",  (CGUIPython::py_send_command), METH_VARARGS, "send command to TextGUI."},
    {"system",  (CGUIPython::py_system), METH_VARARGS, "Execute a shell command."},
    {"get_kernel_matrix",  (CGUIPython::py_get_kernel_matrix), METH_VARARGS, "Execute a shell command."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

PyMODINIT_FUNC initgf(void)
{
	// initialize textgui
	gui=new CTextGUI(0, NULL) ;

    // callback to cleanup at exit
	Py_AtExit(exitgf);

	// initialize callbacks
    (void) Py_InitModule("gf", gfpythonmethods);

	// init Numeric simulation API
	import_array();

	// init Numarray API
	import_libnumarray();
}

void exitgf(void)
{
	CIO::message(M_INFO, "quitting...\n");
	delete gui;
}
#endif
