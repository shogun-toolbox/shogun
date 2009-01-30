#include "lib/config.h"
#include "lib/python.h"

#include <stdio.h>

#ifdef HAVE_PYTHON
void sg_print_message(FILE* target, const char* str)
{
	fprintf(target, "%s", str);
}

void sg_print_warning(FILE* target, const char* str)
{
	if (target==stdout)
		PyErr_Warn(NULL, str);
	else
		fprintf(target, "%s", str);
}

void sg_print_error(FILE* target, const char* str)
{
	if (target==stdout)
		PyErr_SetString(PyExc_RuntimeError, str);
	else
		fprintf(target, "%s", str);
}

void sg_cancel_computations(bool &delayed, bool &immediately)
{
	if (PyErr_CheckSignals())
	{
		SG_SPRINT("\nImmediately return to matlab prompt / Prematurely finish computations / Do nothing (I/P/D)? ");
		char answer=fgetc(stdin);

		if (answer == 'I')
			immediately=true;
		else if (answer == 'P')
		{
			PyErr_Clear();
			delayed=true;
		}
		else
			SG_SPRINT("\n");
	}
}
#endif
