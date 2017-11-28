#include <R.h>
#include <R_ext/Error.h>
#include <R_ext/RS.h>
#include <R_ext/Rdynload.h>
#include <Rdefines.h>
#include <Rembedded.h>
#include <Rinterface.h>
#include <Rinternals.h>

#include <stdio.h>

void sg_global_print_message(FILE* target, const char* str)
{
	if (target == stdout)
		Rprintf((char*)"%s", str);
	else
		fprintf(target, "%s", str);
}

void sg_global_print_warning(FILE* target, const char* str)
{
	if (target == stdout)
		Rprintf((char*)"%s", str);
	else
		fprintf(target, "%s", str);
}

void sg_global_print_error(FILE* target, const char* str)
{
	if (target == stdout)
		Rprintf((char*)"%s", str);
	else
		fprintf(target, "%s", str);
}
