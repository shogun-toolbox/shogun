/* For co-existence with stdio only */

#ifdef __cplusplus
extern "C" {
#endif

#define PERLIO_NOT_STDIO 0
#include <pdlcore.h>

#ifdef __cplusplus
}
#endif


#include <io/SGIO.h>

void sg_global_print_message(FILE* target, const char* str)
{
	fprintf(target, "%s", str);
}

void sg_global_print_warning(FILE* target, const char* str)
{
	SV *err = get_sv("@", GV_ADD);
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
	//SWIG_croak(str);
	else
		fprintf(target, "%s", str);
}

//PTZ121009 used in threads...so cannot access stdin like this...
// why not checking kill stuff???
void sg_global_cancel_computations(bool &delayed, bool &immediately)
{
#if 0
	using namespace shogun;

	dTHX;       /* fetch context */
	//PerlIO_init(pTHX);
	//  PerlIO *f = PerlIO_stdin(); crashes in Perl_csighandler () from /usr/lib/libperl.so.5.14

	if(!f) {return;}
	if(PerlIO_flush(f)) //check signal
	{
		SG_SPRINT("\nImmediately return to matlab prompt / Prematurely finish computations / Do nothing (I/P/D)? ");
		char answer= PerlIO_getc(f);
		if (answer == 'I')
			immediately=true;
		else if (answer == 'P')
		{
			PerlIO_clearerr(f);
			delayed=true;
		}
		else
			SG_SPRINT("\n");
	}
#endif
}
