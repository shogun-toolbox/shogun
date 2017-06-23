#undef _POSIX_C_SOURCE
extern "C" {
#include <Python.h>
}

#include <shogun/io/SGIO.h>
#include <stdio.h>

void sg_global_print_message(FILE* target, const char* str)
{
	fprintf(target, "%s", str);
}

void sg_global_print_warning(FILE* target, const char* str)
{
	if (target == stdout)
	{
		PyGILState_STATE gil = PyGILState_Ensure();
		PyErr_Warn(NULL, str);
		PyGILState_Release(gil);
	}
	else
		fprintf(target, "%s", str);
}

void sg_global_print_error(FILE* target, const char* str)
{
	if (target == stdout)
	{
		PyGILState_STATE gil = PyGILState_Ensure();
		PyErr_SetString(PyExc_RuntimeError, str);
		PyGILState_Release(gil);
	}
	else
		fprintf(target, "%s", str);
}
