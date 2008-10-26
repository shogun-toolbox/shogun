%{
#include "preproc/StringPreProc.h" 
%}

%include "preproc/StringPreProc.h" 

%template(StringUlongPreProc) CStringPreProc<uint64_t>;
%template(StringWordPreProc) CStringPreProc<uint16_t>;
%template(StringBytePreProc) CStringPreProc<uint8_t>;
%template(StringCharPreProc) CStringPreProc<char>;
