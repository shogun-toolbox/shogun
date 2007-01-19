%{
#include "preproc/StringPreProc.h" 
%}

%include "lib/common.i"

%feature("director") CPreProc;
%feature("autodoc","0");

%include "preproc/StringPreProc.h" 

%template(StringUlongPreProc) CStringPreProc<ULONG>;
%template(StringWordPreProc) CStringPreProc<WORD>;
%template(StringBytePreProc) CStringPreProc<BYTE>;
%template(StringCharPreProc) CStringPreProc<CHAR>;
