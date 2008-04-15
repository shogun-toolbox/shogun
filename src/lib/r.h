#include "lib/config.h"

#ifdef HAVE_R

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
}

#endif //__R_H__
#endif //HAVE_R
