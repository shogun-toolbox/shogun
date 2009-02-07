%define DOCSTR
"The `PreProc` module gathers all preprocessors available in the SHOGUN toolkit."
%enddef

%module(docstring=DOCSTR) PreProc
%{
#include "preproc/PreProc.h"
%}

%rename(PreProc) CPreProc;
%feature("autodoc","0");

#ifdef HAVE_DOXYGEN
%include "preproc/PreProc_doxygen.i"
#endif

%include "lib/common.i"
%include "lib/ShogunException.i"
%include "lib/io.i"
%include "base/Version.i"
%include "base/Parallel.i"
%include "base/SGObject.i"
%include "preproc/PreProc.h"

%include "preproc/SimplePreProc.i"
%include "preproc/StringPreProc.i"

%include "preproc/LogPlusOne.i"
%include "preproc/NormDerivativeLem3.i"
%include "preproc/NormOne.i"
%include "preproc/PCACut.i"
%include "preproc/PruneVarSubMean.i"
%include "preproc/SortUlongString.i"
%include "preproc/SortWordString.i"
%include "preproc/SparsePreProc.i"
