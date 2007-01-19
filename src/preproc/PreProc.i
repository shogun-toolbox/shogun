%define DOCSTR
"The `Kernel` module gathers all kernels available in the SHOGUN toolkit."
%enddef

%module(docstring=DOCSTR,directors="1") PreProc
%{
#define SWIG_FILE_WITH_INIT
#include "preproc/PreProc.h" 
%}

%include "lib/common.i"

%feature("director") CPreProc;
%rename(PreProc) CPreProc;
%feature("autodoc","0");

%include "preproc/PreProc.h" 
%include "preproc/StringPreProc.i" 
