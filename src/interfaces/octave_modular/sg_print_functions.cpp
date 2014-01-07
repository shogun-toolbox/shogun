#include <octave/config.h>

#include <octave/ov.h>
#include <octave/defun-dld.h>
#include <octave/error.h>
#include <octave/oct-obj.h>
#include <octave/pager.h>
#include <octave/symtab.h>
#include <octave/variables.h>
#include <octave/Cell.h>

#include <io/SGIO.h>
#include <stdio.h>

void sg_global_print_message(FILE* target, const char* str)
{
	fprintf(target, "%s", str);
}

void sg_global_print_warning(FILE* target, const char* str)
{
	if (target==stdout)
		::warning(str);
	else
		fprintf(target, "%s", str);
}

void sg_global_print_error(FILE* target, const char* str)
{
	fprintf(target, "%s", str);
}

void sg_global_cancel_computations(bool &delayed, bool &immediately)
{
}
