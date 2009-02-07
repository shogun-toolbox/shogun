#ifndef __R_H__
#define __R_H__

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

/* workaround compile bug in R-modular interface */
#if defined(HAVE_R) && !defined(ScalarReal)
#define ScalarReal      Rf_ScalarReal
#endif

#endif //__R_H__
