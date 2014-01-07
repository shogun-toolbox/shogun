extern "C" {
#include <R.h>
#include <Rinternals.h>
#include <Rdefines.h>
#include <R_ext/Rdynload.h>
#include <Rembedded.h>
#include <Rinterface.h>
#include <R_ext/RS.h>
#include <R_ext/Error.h>
}

#include <base/SGObject.h>
#include <stdio.h>

using namespace shogun;

void sg_global_print_message(FILE* target, const char* str)
{
	if (target==stdout)
		Rprintf((char*) "%s", str);
	else
		fprintf(target, "%s", str);
}

void sg_global_print_warning(FILE* target, const char* str)
{
	if (target==stdout)
		Rprintf((char*) "%s", str);
	else
		fprintf(target, "%s", str);
}

void sg_global_print_error(FILE* target, const char* str)
{
	if (target==stdout)
		Rprintf((char*) "%s", str);
	else
		fprintf(target, "%s", str);
}

void sg_global_cancel_computations(bool &delayed, bool &immediately)
{
	/* R_Suicide((char*) "sg stopped by SIGINT\n"); */
}
