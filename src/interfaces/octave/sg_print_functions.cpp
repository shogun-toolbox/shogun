#if ((OCTAVE_MAJOR_VERSION == 4) && (OCTAVE_MINOR_VERSION >= 4))
#include <octave/octave-config.h>
#else
#include <octave/config.h>
#endif

#include <octave/Cell.h>
#include <octave/defun-dld.h>
#include <octave/error.h>
#include <octave/oct-obj.h>
#include <octave/ov.h>
#include <octave/pager.h>
#include <octave/symtab.h>
#include <octave/variables.h>

#include <shogun/io/SGIO.h>
#include <stdio.h>

void sg_global_print_message(FILE* target, const char* str)
{
	fprintf(target, "%s", str);
}

void sg_global_print_warning(FILE* target, const char* str)
{
	if (target == stdout)
		::warning(str);
	else
		fprintf(target, "%s", str);
}

void sg_global_print_error(FILE* target, const char* str)
{
	fprintf(target, "%s", str);
}
