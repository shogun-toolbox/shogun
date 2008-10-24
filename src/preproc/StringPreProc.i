%{
#include "preproc/StringPreProc.h" 
%}

%include "preproc/StringPreProc.h" 

%template(StringUlongPreProc) CStringPreProc<ULONG>;
%template(StringWordPreProc) CStringPreProc<WORD>;
%template(StringBytePreProc) CStringPreProc<uint8_t>;
%template(StringCharPreProc) CStringPreProc<char>;
