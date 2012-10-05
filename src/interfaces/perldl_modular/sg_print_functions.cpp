//PTZ120922 why?!
//#undef _POSIX_C_SOURCE


#define PERLIO_NOT_STDIO 0    /* For co-existence with stdio only */
  //#include <perlio.h>           /* Usually via #include <perl.h> */

#ifdef __cplusplus
extern "C" {
#endif

#include <pdlcore.h>

#ifdef __cplusplus
}
#endif


#include <shogun/io/SGIO.h>

void sg_global_print_message(FILE* target, const char* str)
{
  fprintf(target, "%s", str);
}
//from perlrun.swg SWIG_croak(str);
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

void sg_global_cancel_computations(bool &delayed, bool &immediately)
{
  using namespace shogun;
  PerlIO *perlio = PerlIO_stdin();
  if(PerlIO_flush(perlio)) //check signal
    {
      SG_SPRINT("\nImmediately return to matlab prompt / Prematurely finish computations / Do nothing (I/P/D)? ");
      char answer= PerlIO_getc(perlio);   
      if (answer == 'I')
	immediately=true;
      else if (answer == 'P')
	{
	  PerlIO_clearerr(perlio);
	  delayed=true;
	}
      else
	SG_SPRINT("\n");
    }
}
