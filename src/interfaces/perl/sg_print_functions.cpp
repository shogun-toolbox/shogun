/* For co-existence with stdio only */

#ifdef __cplusplus
extern "C" {
#endif

#define PERLIO_NOT_STDIO 0
#include <pdlcore.h>

#ifdef __cplusplus
}
#endif

#include <shogun/io/SGIO.h>

void sg_global_print_message(FILE* target, const char* str)
{
	fprintf(target, "%s", str);
}

void sg_global_print_warning(FILE* target, const char* str)
{
	SV* err = get_sv("@", GV_ADD);
	if (target == stdout)
	{
		if (sv_isobject(err))
			pdl_warn(0);
		else
			croak("%s", SvPV_nolen(err));
	}
	else
		fprintf(target, "%s", str);
}

void sg_global_print_error(FILE* target, const char* str)
{
	if (target == stdout) // PerlIO_stdout()) //"ERRSV" ($@)
		croak(str);
	// SWIG_croak(str);
	else
		fprintf(target, "%s", str);
}
