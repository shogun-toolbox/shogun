#include <shogun/lib/config.h>

#include "lib/r.h"

#include <stdio.h>

#ifdef HAVE_R
void sg_print_message(FILE* target, const char* str)
{
	if (target==stdout)
		Rprintf((char*) "%s", str);
	else
		fprintf(target, "%s", str);
}

void sg_print_warning(FILE* target, const char* str)
{
	if (target==stdout)
		Rprintf((char*) "%s", str);
	else
		fprintf(target, "%s", str);
}

void sg_print_error(FILE* target, const char* str)
{
	if (target==stdout)
		Rprintf((char*) "%s", str);
	else
		fprintf(target, "%s", str);
}

void sg_cancel_computations(bool &delayed, bool &immediately)
{
			//R_Suicide((char*) "sg stopped by SIGINT\n");
}
#endif

