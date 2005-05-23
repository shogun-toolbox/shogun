#include "lib/config.h"

#ifdef HAVE_PYTHON
#include "guilib/GUIPython.h"
#include "gui/Python.h"
#include "gui/TextGUI.h"

static CGUIPython gfpy;
extern CTextGUI* gui;

static PyMethodDef gfpythonmethods[] = {
    {"send_command",  (CGUIPython::py_send_command), METH_VARARGS, "Send command to TextGUI."},
    {"system",  (CGUIPython::py_system), METH_VARARGS, "Execute a shell command."},
    {"set_features",  (CGUIPython::py_set_features), METH_VARARGS, "Set a feature object."},
    {"get_kernel_matrix",  (CGUIPython::py_get_kernel_matrix), METH_VARARGS, "Get the kernel matrix."},
    {"set_kernel_matrix",  (CGUIPython::py_set_kernel_matrix), METH_VARARGS, "Set the kernel matrix."},
    {"test",  (CGUIPython::py_test), METH_VARARGS, "Test."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

PyMODINIT_FUNC initgf(void)
{
	// initialize textgui
	gui=new CTextGUI(0, NULL) ;

    // callback to cleanup at exit
	Py_AtExit(exitgf);

	// initialize callbacks
    Py_InitModule("gf", gfpythonmethods);
}

void exitgf(void)
{
	CIO::message(M_INFO, "quitting...\n");
	delete gui;
}
#endif
