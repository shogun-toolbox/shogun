#include "lib/config.h"
#include "lib/matlab.h"

#include <stdio.h>

#ifdef HAVE_MATLAB
void sg_print_message(FILE* target, const char* str)
{
	if (target==stdout)
		mexPrintf("%s", str);
	else
		fprintf(target, "%s", str);
}

void sg_print_warning(FILE* target, const char* str)
{
	if (target==stdout)
		mexWarnMsgTxt(str);
	else
		fprintf(target, "%s", str);
}

void sg_print_error(FILE* target, const char* str)
{
	if (target==stdout)
		mexPrintf("%s", str);
	else
		fprintf(target, "%s", str);
}

void sg_cancel_computations(bool &delayed, bool &immediately)
{
}
#endif
