%{
#include "preproc/StringPreProc.h" 
%}

%include "preproc/StringPreProc.h" 

%template(StringUlongPreProc) CStringPreProc<ULONG>;
%template(StringWordPreProc) CStringPreProc<WORD>;
%template(StringBytePreProc) CStringPreProc<BYTE>;
%template(StringCharPreProc) CStringPreProc<char>;
