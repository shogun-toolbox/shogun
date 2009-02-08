%define DOCSTR
"The `PreProc` module gathers all preprocessors available in the SHOGUN toolkit."
%enddef

%module(docstring=DOCSTR) PreProc
%{
#include <shogun/preproc/PreProc.h>
%}

%rename(PreProc) CPreProc;
%feature("autodoc","0");

#ifdef HAVE_DOXYGEN
%include "PreProc_doxygen.i"
#endif

%include "common.i"
%include "ShogunException.i"
%include "io.i"
%include "Version.i"
%include "Parallel.i"
%include "SGObject.i"
%include <shogun/preproc/PreProc.h>

%include "SimplePreProc.i"
%include "StringPreProc.i"

%include "LogPlusOne.i"
%include "NormDerivativeLem3.i"
%include "NormOne.i"
%include "PCACut.i"
%include "PruneVarSubMean.i"
%include "SortUlongString.i"
%include "SortWordString.i"
%include "SparsePreProc.i"
